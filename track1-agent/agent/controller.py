# SPDX-License-Identifier: Apache-2.0
"""The agent state machine.

    ingest -> profile -> contract -> PLAN (think) -> CODE (no-think, continuation on truncation)
      -> loop { assemble -> run in stage dir -> validate -> keep-best/promote
                -> REPAIR (think, SEARCH/REPLACE) | REWRITE on stagnation }
      -> REVIEW (think) of clean outputs -> apply fixes -> re-validate (best is kept)
      -> sanitize output tree -> exit 0

The controller, not the model, decides when to stop. Two budgets bind: House requests and wall
clock (the per-unit card ceiling and, more tightly, the shared development stage clock).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import time
from dataclasses import dataclass, field

from . import casebook, codeio, prompts
from .compare import compare_dirs
from .transform import stable_sorts
from .apidoctor import hints as api_hints
from .config import Config
from .contract import Contract, extract
from .finalize import STAGE_NAME, promote, sanitize, write_skeleton

B_STAGE = ".agent_stage_b"
from .llm import BudgetExhausted, HouseClient, LLMResult
from .log import log, save_debug
from .profile import build_profile, inventory
from .runner import RunResult, run_solution
from .task import Task, card_summary, load_task
from .validate import Report, validate


@dataclass
class Attempt:
    n: int
    code: str
    origin: str
    run: RunResult | None = None
    report: Report = field(default_factory=Report)


class Controller:
    def __init__(self, task_dir: str, out_dir: str, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.t0 = time.time()
        self.task_dir = task_dir
        self.out = pathlib.Path(out_dir).absolute()
        self.work = pathlib.Path(self.cfg.work_root).absolute()
        self.llm = HouseClient(self.cfg)
        self.attempts: list[Attempt] = []
        self.best: Attempt | None = None
        self.plan = ""
        self.history: list[str] = []
        self.deadline = self.t0 + self.cfg.max_unit_seconds

    # ----------------------------------------------------------------- time helpers
    def left(self) -> float:
        return self.deadline - time.time()

    def exec_budget(self) -> float:
        return max(10.0, min(self.cfg.exec_timeout_cap, self.left() - 40.0))

    # ----------------------------------------------------------------- main
    def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.task: Task = load_task(self.task_dir, str(self.out), use_checks=self.cfg.use_checks_dir)
        card_cap = self.task.timeout_sec - self.cfg.card_margin_seconds
        self.deadline = self.t0 + max(60.0, min(card_cap, self.cfg.max_unit_seconds))
        log(f"unit={self.task.unit_id} category={self.task.category} timeout={self.task.timeout_sec:.0f}s "
            f"budget={self.deadline - self.t0:.0f}s files={len(self.task.files)} checks={'yes' if self.task.checks_text else 'no'}")
        self.contract: Contract = extract(self.task)
        self.profile = build_profile(self.task)
        self.inv = inventory(self.task)
        self.examples = casebook.retrieve(self.task) if self.cfg.use_casebook else ""
        self.ctx_full = prompts.task_block(
            self.task.instruction, card_summary(self.task), self.inv, self.profile,
            self.contract.describe(), self.task.checks_text[:30000], self.examples,
        )
        self.ctx_compact = prompts.task_block(
            self.task.instruction, card_summary(self.task), self.inv, self.profile,
            self.contract.describe(), self.task.checks_text[:20000], "", compact=True,
        )
        save_debug("context.txt", self.ctx_full)
        save_debug("contract.txt", self.contract.describe())
        if not self.llm.available:
            log("House model not configured: writing skeleton outputs only")
            return
        try:
            self._solve()
        except BudgetExhausted:
            log("request budget exhausted")

    def finish(self) -> None:
        try:
            if self.best is None or not any(p.name != STAGE_NAME for p in self.out.iterdir()):
                write_skeleton(self.out, self.contract)
        except Exception as e:  # noqa: BLE001
            log(f"skeleton failed: {e}")
        sanitize(self.out, self.task.canaries if hasattr(self, "task") else set())
        summary = {
            "unit": getattr(getattr(self, "task", None), "unit_id", "?"),
            "seconds": round(time.time() - self.t0, 1),
            "house_requests": self.llm.charged,
            "attempts": [
                {"n": a.n, "origin": a.origin, "score": a.report.score, "errors": a.report.errors[:5],
                 "run_s": round(a.run.seconds, 1) if a.run else None}
                for a in self.attempts
            ],
            "best": self.best.n if self.best else None,
            "calls": self.llm.calls,
        }
        save_debug("summary.json", json.dumps(summary, indent=1, default=str))
        log(f"done: {json.dumps({k: summary[k] for k in ('seconds', 'house_requests', 'best')})}")

    # ----------------------------------------------------------------- model helpers
    def _msgs(self, user: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        m = [{"role": "system", "content": prompts.SYSTEM}]
        m += history or []
        m.append({"role": "user", "content": user})
        return m

    def _ask(self, role: str, user: str, *, thinking: bool, history: list[dict[str, str]] | None = None,
             max_tokens: int | None = None) -> LLMResult:
        budget = None
        if thinking:
            budget = self.cfg.reasoning_budget_plan if role == "plan" else self.cfg.reasoning_budget_fix
        return self.llm.chat(self._msgs(user, history), role=role, thinking=thinking, max_tokens=max_tokens,
                             deadline=self.deadline, reasoning_budget=budget)

    # ----------------------------------------------------------------- phases
    def _solve(self) -> None:
        self._make_plan()
        code = self._write_code(fresh_note="")
        if not code and self.llm.remaining > 3 and self.left() > 150:
            # both code calls failed (often a transient network/stream problem): pause, try once more
            log("no code from the first code calls; retrying after a short pause")
            time.sleep(10)
            code = self._write_code(fresh_note="")
        if not code:
            return
        self._loop(code, "initial")

    def _make_plan(self) -> None:
        res = self._ask("plan", self.ctx_full + "\n\n" + prompts.KNOWLEDGE + "\n\n" + prompts.PLAN_FORMAT, thinking=True)
        plan = res.content
        if res.ok and len(plan) < 200 and res.reasoning:
            # thinking consumed the output budget: salvage the analysis as plan notes
            plan = "(planning notes)\n" + res.reasoning[-7000:]
        self.plan = plan.strip()
        save_debug("plan.md", self.plan)
        log(f"plan: {len(self.plan)} chars, finish={res.finish_reason}, ok={res.ok}")

    def _write_code(self, fresh_note: str, independent: bool = False) -> str:
        rules = prompts.CODE_RULES.format(exec_budget=int(self.exec_budget()), cpus=self._threads())
        user = self.ctx_full + "\n\n" + prompts.KNOWLEDGE
        if independent:
            user += "\n\n" + prompts.INDEPENDENT_NOTE
        elif self.plan:
            user += "\n\n=== IMPLEMENTATION PLAN ===\n" + self.plan
        if fresh_note:
            user += "\n\n=== LESSONS FROM FAILED ATTEMPTS (avoid these) ===\n" + fresh_note
        user += "\n\n" + rules
        res = self._ask("code_b" if independent else "code", user, thinking=False)
        if (not res.ok or not res.content.strip()) and self.llm.remaining > 3 and self.left() > 120:
            log("code call returned nothing; retrying once")
            res = self._ask("code", user, thinking=False)
        if not res.ok:
            return ""
        text = res.content
        conts = 0
        while res.truncated and conts < 2 and self.llm.remaining > 3 and self.left() > 90:
            conts += 1
            hist = [{"role": "user", "content": user}, {"role": "assistant", "content": text}]
            res = self._ask(
                "code_cont",
                "Your reply was cut off by the output limit. Continue the program EXACTLY from the last "
                "character you wrote. Output only the continuation (no repetition, no explanation, no code fence).",
                thinking=False,
                history=hist,
            )
            if not res.ok:
                break
            text = codeio.join_continuation(text, res.content)
            text = "```python\n" + text + "\n```" if "```" not in text else text
        code = codeio.extract_code(text)
        if not code:
            log("no code extracted from model reply")
        return codeio.strip_prelude(code)

    def _threads(self) -> int:
        n = self.task.cpus or os.cpu_count() or 4
        return max(1, min(16, n))

    # ----------------------------------------------------------------- attempt execution
    def _assemble(self, code: str, stage: pathlib.Path) -> str:
        """Model code -> executable file. Every edit preserves the model's line numbering so the
        traceback lines shown to the repair model match the numbered listing it sees."""
        code = codeio.strip_prelude(code)
        code = codeio.remap_paths(code, self.task, str(stage))
        code, n_sorts = stable_sorts(code)
        if n_sorts:
            log(f"hardening: {n_sorts} sort call(s) made stable")
        lines = code.split("\n")
        future = [l for l in lines if l.startswith("from __future__")]
        body = "\n".join("# (from __future__ import moved to top)" if l.startswith("from __future__") else l for l in lines)
        head = "\n".join(future) + ("\n" if future else "") + codeio.build_prelude(self.task, str(stage)) + "\n"
        self._line_offset = head.count("\n")
        return head + body + "\n"

    def _fix_lines(self, text: str, script_name: str) -> str:
        """Translate traceback line numbers of the assembled file back to the model's numbering."""
        off = getattr(self, "_line_offset", 0)
        if not off or not text:
            return text

        def repl(m: re.Match[str]) -> str:
            n = int(m.group(2)) - off
            return f"{m.group(1)}{n if n > 0 else m.group(2)}"

        return re.sub(r'(File "[^"]*' + re.escape(script_name) + r'", line )(\d+)', repl, text)

    def _fix_syntax(self, code: str) -> str:
        """Cheap, no-thinking repair of compile errors on a local window (up to 3 rounds)."""
        for _ in range(3):
            try:
                compile(code, "solution.py", "exec")
                return code
            except SyntaxError as e:
                if self.llm.remaining <= 2 or self.left() < 60:
                    return code
                lines = code.split("\n")
                n = max(1, min(e.lineno or 1, len(lines)))
                a, b = max(1, n - 25), min(len(lines), n + 12)
                window = "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(a, b + 1))
                err = f"SyntaxError: {e.msg} (line {e.lineno}): {(e.text or '').strip()[:160]}"
                res = self._ask("syntax_fix", prompts.SYNTAX_FIX.format(error=err, a=a, b=b, window=window),
                                thinking=False, max_tokens=1500)
                if not res.ok:
                    return code
                new, applied, _ = codeio.apply_patches(code, res.content)
                log(f"syntax fix: {err[:100]} -> {applied} edit(s)")
                if not applied:
                    return code
                code = new
        return code

    def _execute(self, code: str, origin: str, promote_ok: bool = True, keep_stage_as: str = "") -> Attempt:
        code = self._fix_syntax(code)
        n = len(self.attempts) + 1
        att = Attempt(n=n, code=code, origin=origin)
        stage = self.out / STAGE_NAME
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)
        full = self._assemble(code, stage)
        script = self.work / f"solution_{n}.py"
        script.write_text(full, encoding="utf-8")
        save_debug(f"solution_{n}.py", full)
        err = codeio.syntax_error(full)
        issues = codeio.static_issues(code)
        if err:
            att.run = RunResult(1, "", err, 0.0, False)
        elif any(i.startswith("forbidden") for i in issues):
            att.run = RunResult(1, "", "static check refused the program: " + "; ".join(issues), 0.0, False)
        else:
            att.run = run_solution(script, self.work, timeout=self.exec_budget(), threads=self._threads())
            att.run.stderr = self._fix_lines(att.run.stderr, script.name)
        att.report = validate(stage, self.contract, att.run, [])
        if att.run.stderr and att.run.returncode != 0:
            att.report.errors.append("traceback tail:\n" + att.run.stderr[-2500:])
            facts = api_hints(att.run.stderr[-4000:])
            if facts:
                att.report.errors.insert(1, "API FACTS (introspected from the installed libraries; trust these, "
                                            "not memory):\n" + facts)
                self.history.append("API facts: " + facts[:400])
                log(f"api doctor: {facts[:160]}")
        for i in issues:
            att.report.warnings.append(i)
        self.attempts.append(att)
        log(f"attempt {n} ({origin}): rc={att.run.returncode} t={att.run.seconds:.1f}s score={att.report.score} "
            f"files={att.report.files[:8]} err0={(att.report.errors[:1] or [''])[0][:160]!r}")
        better = self.best is None or att.report.score <= self.best.report.score
        if promote_ok and better and att.report.files:
            self.best = att
            promote(stage, self.out)
        if keep_stage_as:
            kept = self.out / keep_stage_as
            shutil.rmtree(kept, ignore_errors=True)
            stage.rename(kept)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        return att

    # ----------------------------------------------------------------- repair / review loop
    def _loop(self, code: str, origin: str) -> None:
        repairs = 0
        rewrites = 0
        reviews = 0
        dual_done = not self.cfg.dual
        signatures: list[str] = []
        att = self._execute(code, origin)
        while True:
            if self.left() < 45 or self.llm.remaining <= 0:
                return
            rep = att.report
            if rep.clean:
                spent = time.time() - self.t0
                budget = max(1.0, self.deadline - self.t0)
                # the independent re-implementation is optional: only when the unit is on schedule
                if not dual_done and self.llm.remaining >= 3 and self.left() > 200 and spent < 0.75 * budget:
                    dual_done = True
                    outcome, nxt = self._differential(att)
                    if outcome == "agree":
                        return  # two independent implementations agree: high confidence, stop
                    if outcome == "adjudicated":
                        reviews = self.cfg.review_passes  # the adjudication was the review
                        if nxt is None:
                            return
                        att = nxt
                        continue
                if reviews >= self.cfg.review_passes or self.left() < 150 or self.llm.remaining < 2:
                    return
                reviews += 1
                new_code = self._review(att)
                if not new_code or new_code == att.code:
                    return
                att = self._execute(new_code, f"review{reviews}")
                continue
            # failing attempt: repair or rewrite
            sig = _signature(rep)
            signatures.append(sig)
            stagnating = len(signatures) >= 3 and len(set(signatures[-3:])) == 1
            if repairs >= self.cfg.max_repairs:
                return
            if stagnating and rewrites < 1 and self.llm.remaining >= 3 and self.left() > 180:
                rewrites += 1
                note = "\n".join(self.history[-6:])
                new_code = self._write_code(fresh_note=note)
                signatures.clear()
            else:
                repairs += 1
                new_code = self._repair(att)
                if not new_code and rewrites < 2 and self.llm.remaining >= 3 and self.left() > 180:
                    # the repair produced nothing usable: fall back to a fresh implementation
                    rewrites += 1
                    stagnating = True
                    new_code = self._write_code(fresh_note="\n".join(self.history[-6:]))
            if not new_code:
                return
            att = self._execute(new_code, "rewrite" if stagnating else f"repair{repairs}")

    # ----------------------------------------------------------------- differential self-consistency
    def _differential(self, att: Attempt) -> tuple[str, Attempt | None]:
        """Write an independent implementation B, run it, compare with A (already promoted)."""
        code_b = self._write_code(fresh_note="", independent=True)
        if not code_b:
            return "skipped", None
        att_b = self._execute(code_b, "independent", promote_ok=False, keep_stage_as=B_STAGE)
        stage_b = self.out / B_STAGE
        try:
            if not att_b.report.clean:
                log("independent implementation not clean; falling back to review")
                return "skipped", None
            names = sorted(set(self.contract.strong_files) | set(att.report.files))
            diffs = compare_dirs(self.out, stage_b, names)
            save_debug("differential.txt", "\n".join(diffs) or "AGREE")
            if not diffs:
                log("differential check: independent implementations AGREE")
                return "agree", None
            log(f"differential check: {len(diffs)} disagreement(s); adjudicating")
            choice, edits_code = self._adjudicate(att, att_b, diffs)
            chosen = att
            if choice == "B":
                promote(stage_b, self.out)
                self.best = att_b
                chosen = att_b
            if edits_code and edits_code != chosen.code:
                return "adjudicated", self._execute(edits_code, f"adjudicated-{choice}")
            return "adjudicated", None
        finally:
            shutil.rmtree(stage_b, ignore_errors=True)

    def _adjudicate(self, a: Attempt, b: Attempt, diffs: list[str]) -> tuple[str, str]:
        user = (
            self.ctx_compact
            + "\n\n=== PROGRAM A (line-numbered) ===\n" + codeio.numbered(a.code, 500)
            + "\n\n=== PROGRAM B (line-numbered, written independently) ===\n" + codeio.numbered(b.code, 500)
            + "\n\n=== OUTPUT DISAGREEMENTS BETWEEN A AND B ===\n" + "\n".join(diffs[:40])
            + "\n\n=== A OUTPUT PREVIEW ===\n" + (a.report.preview[:2500] or "(none)")
            + "\n\n=== B OUTPUT PREVIEW ===\n" + (b.report.preview[:2500] or "(none)")
            + "\n\n" + prompts.ADJUDICATE_RULES
        )
        res = self._ask("adjudicate", user, thinking=True, max_tokens=3600)
        if not res.ok:
            return "A", ""
        m = re.search(r"VERDICT:\s*([AB])\b", res.content)
        choice = m.group(1) if m else "A"
        base = b.code if choice == "B" else a.code
        new, applied, _ = codeio.apply_patches(base, res.content)
        log(f"adjudication: choose {choice}; {applied} edit(s)")
        return choice, new if applied else ""

    def _exec_block(self, att: Attempt) -> str:
        r = att.run
        if r is None:
            return ""
        return (
            f"exit_code={r.returncode} timed_out={r.timed_out} seconds={r.seconds:.1f}\n"
            f"--- stdout (tail) ---\n{r.stdout[-3000:]}\n--- stderr (tail) ---\n{r.stderr[-3500:]}"
        )

    def _repair(self, att: Attempt) -> str:
        self.history.append(f"attempt {att.n}: " + "; ".join(e.splitlines()[0][:200] for e in att.report.errors[:4]))
        user = (
            self.ctx_compact
            + ("\n\n=== IMPLEMENTATION PLAN ===\n" + self.plan[:6000] if self.plan else "")
            + "\n\n=== CURRENT PROGRAM (line-numbered; the agent prepends TASK_DIR/DATA_DIR/OUTPUT_DIR/write_json) ===\n"
            + codeio.numbered(att.code)
            + "\n\n=== EXECUTION RESULT ===\n" + self._exec_block(att)
            + "\n\n=== AUTOMATIC VALIDATION OF THE OUTPUTS ===\n" + att.report.summary()
            + ("\n\n=== OUTPUT PREVIEW ===\n" + att.report.preview[:4000] if att.report.preview else "")
            + ("\n\n=== EARLIER FAILED ATTEMPTS ===\n" + "\n".join(self.history[-5:]) if len(self.history) > 1 else "")
            + "\n\n" + prompts.REPAIR_RULES
        )
        res = self._ask("repair", user, thinking=True)
        return self._apply_reply(att.code, res, user)

    def _review(self, att: Attempt) -> str:
        user = (
            self.ctx_compact
            + ("\n\n=== IMPLEMENTATION PLAN ===\n" + self.plan[:6000] if self.plan else "")
            + "\n\n=== CURRENT PROGRAM (line-numbered) ===\n" + codeio.numbered(att.code)
            + "\n\n=== EXECUTION RESULT ===\n" + self._exec_block(att)
            + "\n\n=== OUTPUT PREVIEW ===\n" + (att.report.preview or "(none)")
            + "\n\n=== AUTOMATIC VALIDATION ===\n" + att.report.summary()
            + "\n\n" + prompts.REVIEW_RULES
        )
        res = self._ask("review", user, thinking=True, max_tokens=3200)
        if not res.ok:
            return ""
        verdict = re.search(r"VERDICT:\s*(PASS|FIX)", res.content, re.I)
        if verdict and verdict.group(1).upper() == "PASS":
            log("review: PASS")
            return ""
        if not res.content.strip() and res.truncated:
            log("review: thinking exhausted the budget, keeping current outputs")
            return ""
        return self._apply_reply(att.code, res, user, allow_rewrite=False)

    def _apply_reply(self, code: str, res: LLMResult, user: str, allow_rewrite: bool = True) -> str:
        if not res.ok:
            return ""
        text = res.content
        if not text.strip() and res.reasoning and self.llm.remaining > 1 and self.left() > 90:
            # thinking ran out of tokens before the answer: ask for the edits without thinking
            hist = [{"role": "user", "content": user},
                    {"role": "assistant", "content": "Analysis so far:\n" + res.reasoning[-5000:]}]
            res2 = self._ask("patch_salvage", "Now output ONLY the SEARCH/REPLACE edits (or the full program) "
                             "implementing your analysis, in the required format.", thinking=False, history=hist)
            if res2.ok:
                text = res2.content
        new, applied, failed = codeio.apply_patches(code, text)
        if applied:
            log(f"applied {applied} edit(s); {len(failed)} failed")
            if failed:
                self.history.append("edit failed to apply: " + failed[0][:120])
            return new
        full = codeio.extract_code(text)
        if full and allow_rewrite and len(full) > 0.4 * len(code):
            log("model returned a full program")
            return codeio.strip_prelude(full)
        if full and not allow_rewrite and len(full) > 0.6 * len(code):
            return codeio.strip_prelude(full)
        if failed:
            log(f"no edit applied ({len(failed)} SEARCH blocks did not match)")
            self.history.append("previous SEARCH blocks did not match the program text; copy lines exactly")
        return ""


def _signature(rep: Report) -> str:
    first = rep.errors[0] if rep.errors else ""
    first = re.sub(r"\d+(\.\d+)?", "#", first.splitlines()[0] if first else "")
    return first[:120]
