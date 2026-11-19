# SPDX-License-Identifier: Apache-2.0
"""House-model client.

Contract (hub docs/HOUSE-MODEL.md):
  POST $MODEL_ENDPOINT/v1/chat/completions, Authorization: Bearer $MODEL_TOKEN, model=$MODEL_NAME.
  Proxy variables are injected and read automatically by urllib (never hardcode a proxy).
  25 admitted requests per unit; <= 4000 output tokens per request; admitted failures and
  retries are charged, 401/403 refusals are not. No vendor-side tools, ever.

The model (Nemotron-3-Super) thinks by default and reasoning tokens share the 4000-token
output cap, so callers choose thinking per request. If thinking exhausts the cap and no
answer is produced, the reasoning text is returned as a salvageable fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .log import log, save_debug

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    error: str = ""
    seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ("length", "interrupted")


class BudgetExhausted(Exception):
    pass


class _StopStream(Exception):
    pass


def _loop_period(lines: list[str]) -> int:
    """Return k (1..3) if the last lines are one block of k lines repeated >= 8 times, else 0."""
    for k in (1, 2, 3):
        need = 8 * k
        if len(lines) < need:
            continue
        tail = lines[-need:]
        block = tail[:k]
        if any(l.strip() for l in block) and all(tail[i] == block[i % k] for i in range(need)):
            return k
    return 0


def _looping(text: str) -> bool:
    lines = [l.rstrip() for l in text.rstrip().split("\n")[:-1]]  # ignore the partial last line
    return _loop_period(lines) > 0


def _trim_loop(text: str) -> str:
    """Remove a trailing degenerate repetition, keeping one copy of the repeated block."""
    lines = text.rstrip().split("\n")
    k = _loop_period([l.rstrip() for l in lines[:-1]])
    if not k:
        return text
    body = lines[:-1]
    block = [l.rstrip() for l in body[-k:]]
    while len(body) >= 2 * k and [l.rstrip() for l in body[-2 * k:-k]] == block:
        body = body[:-k]
    return "\n".join(body) + "\n"


def _never_sent(e: BaseException) -> bool:
    """True when the failure happened before any byte reached the server (not admitted)."""
    import socket

    reason = getattr(e, "reason", e)  # urllib wraps socket errors in URLError.reason
    if isinstance(reason, (socket.gaierror, ConnectionRefusedError)):
        return True
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in (101, 113, 111):  # unreachable/refused
        return True
    return False


class HouseClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
        self.url = endpoint.rstrip("/") + "/v1/chat/completions" if endpoint else ""
        # A deployment might hand us an origin that already carries /v1; never double it.
        if endpoint.rstrip("/").endswith("/v1"):
            self.url = endpoint.rstrip("/") + "/chat/completions"
        self.token = os.environ.get("MODEL_TOKEN", "")
        self.model = os.environ.get("MODEL_NAME", "") or "house"
        self.charged = 0  # requests that may have been admitted (conservative)
        self.calls: list[dict[str, Any]] = []
        self.fatal_auth = False
        # 0 full body; 1 drop seed/top_p/n/stream_options; 2 also chat_template_kwargs; 3 also no streaming
        self.body_level = 0

    # ------------------------------------------------------------------ budget
    @property
    def available(self) -> bool:
        return bool(self.url) and not self.fatal_auth

    @property
    def remaining(self) -> int:
        return max(0, min(self.cfg.request_cap, self.cfg.hard_request_cap) - self.charged)

    # ------------------------------------------------------------------ call
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str,
        thinking: bool,
        max_tokens: int | None = None,
        deadline: float | None = None,
        reasoning_budget: int | None = None,
    ) -> LLMResult:
        if not self.available:
            return LLMResult(ok=False, error="house model not configured")
        if self.remaining <= 0:
            raise BudgetExhausted()
        max_tokens = min(max_tokens or self.cfg.max_output_tokens, self.cfg.max_output_tokens)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.cfg.temperature_think if thinking else self.cfg.temperature_code,
            "top_p": self.cfg.top_p,
            "seed": self.cfg.seed,
            "n": 1,
            "stream": self.cfg.stream,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if self.cfg.stream:
            body["stream_options"] = {"include_usage": True}
        if thinking and reasoning_budget:
            body["chat_template_kwargs"]["reasoning_budget"] = int(reasoning_budget)
        body = self._reduce(body)
        data = json.dumps(body).encode("utf-8")
        attempts = 0
        preconnect_waits = 0
        waited = 0.0
        last_err = ""
        while attempts < 3 and self.remaining > 0:
            attempts += 1
            timeout = self.cfg.http_timeout
            if deadline is not None:
                left = deadline - time.time()
                if left < 20:
                    return LLMResult(ok=False, error="no time left for a model call")
                timeout = max(20.0, min(timeout, left - 5))
            req = urllib.request.Request(
                self.url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "text/event-stream" if body.get("stream") else "application/json",
                },
            )
            t0 = time.time()
            self.charged += 1  # charge before sending: an admitted request is never refunded
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if body.get("stream"):
                        res = self._read_stream(resp, deadline)
                    else:
                        res = self._parse(resp.read().decode("utf-8", "replace"))
                if res.ok and not res.content and not res.reasoning and res.finish_reason not in ("stop", "length"):
                    raise ConnectionError("empty response (stream dropped before any token)")
                res.seconds = time.time() - t0
                self._record(role, thinking, res, messages)
                return res
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001
                    pass
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (401, 403):
                    self.charged -= 1  # refused before admission: not charged
                    self.fatal_auth = True
                    break
                if e.code in (400, 422) and self.body_level < 3:
                    # an optional field was rejected (refused before admission, not charged):
                    # drop optional fields step by step and keep the reduced shape for this unit
                    self.charged -= 1
                    self.body_level += 1
                    body = self._reduce(body)
                    data = json.dumps(body).encode("utf-8")
                    log(f"House rejected the request ({last_err[:120]}); body level -> {self.body_level}")
                    continue
                if e.code in (400, 404, 405, 413, 422):
                    break  # our request is wrong; retrying the same bytes cannot help
                time.sleep(min(2.0 * attempts, 6.0))
            except Exception as e:  # noqa: BLE001 - timeouts, resets, DNS
                last_err = f"{type(e).__name__}: {e}"
                if _never_sent(e):
                    # DNS failure / connection refused / unreachable: the request never reached the
                    # route, so it was not admitted. Do not charge; wait it out within the deadline.
                    self.charged -= 1
                    attempts -= 1
                    preconnect_waits += 1
                    wait = min(30.0, 3.0 * preconnect_waits)
                    waited += wait
                    if waited > 90.0 or (deadline is not None and time.time() + wait > deadline - 30):
                        break
                    log(f"network not reachable ({last_err[:80]}); retry in {wait:.0f}s (not charged)")
                    time.sleep(wait)
                    continue
                time.sleep(min(2.0 * attempts, 6.0))
        res = LLMResult(ok=False, error=last_err or "budget exhausted")
        self._record(role, thinking, res, messages)
        return res

    # ------------------------------------------------------------------ helpers
    def _reduce(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.body_level == 0:
            return body
        keep = ["model", "messages", "max_tokens", "temperature", "stream"]
        if self.body_level == 1:
            keep.append("chat_template_kwargs")
        out = {k: body[k] for k in keep if k in body}
        if self.body_level >= 3:
            out["stream"] = False
        return out

    @staticmethod
    def _read_stream(resp: Any, deadline: float | None) -> LLMResult:
        """Accumulate an OpenAI-style SSE stream (content + reasoning deltas)."""
        content: list[str] = []
        reasoning: list[str] = []
        finish = ""
        usage: dict[str, Any] = {}
        plain: list[str] = []  # server ignored stream=true and sent one JSON body
        saw_sse = False
        n_chunks = 0
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    if not saw_sse and line:
                        plain.append(line)
                    continue
                saw_sse = True
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        content.append(d["content"])
                        n_chunks += 1
                        if n_chunks % 40 == 0 and _looping("".join(content[-400:])):
                            finish = "repetition"
                            log("stream: degenerate repetition detected; stopping early")
                            raise _StopStream()
                    r = d.get("reasoning_content") or d.get("reasoning")
                    if r:
                        reasoning.append(r)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                if deadline is not None and time.time() > deadline:
                    finish = finish or "interrupted"
                    break
        except _StopStream:
            # keep everything before the loop: cut the repeated tail
            content = [_trim_loop("".join(content))]
        except Exception as e:  # noqa: BLE001 - connection dropped mid-stream: keep what arrived
            if not content and not reasoning:
                raise
            finish = finish or "interrupted"
            log(f"stream interrupted after {sum(map(len, content))} chars: {type(e).__name__}")
        if not saw_sse and plain:
            return HouseClient._parse("\n".join(plain))
        text = "".join(content)
        res = HouseClient._parse(json.dumps({"choices": [{"finish_reason": finish, "message": {
            "content": text, "reasoning_content": "".join(reasoning)}}], "usage": usage}))
        return res

    @staticmethod
    def _parse(raw: str) -> LLMResult:
        try:
            obj = json.loads(raw)
            choice = obj["choices"][0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if "<think>" in content:
                m = re.search(r"<think>(.*?)(</think>|$)", content, re.S)
                if m and not reasoning:
                    reasoning = m.group(1)
                content = _THINK_RE.sub("", content)
                if "<think>" in content:  # unterminated thinking: everything is reasoning
                    content = ""
            if "</think>" in content:  # template emitted only the closing tag
                head, _, content = content.partition("</think>")
                reasoning = reasoning or head
            return LLMResult(
                ok=True,
                content=content.strip(),
                reasoning=reasoning.strip(),
                finish_reason=choice.get("finish_reason") or "",
                usage=obj.get("usage") or {},
            )
        except Exception as e:  # noqa: BLE001
            return LLMResult(ok=False, error=f"unparseable response: {e}: {raw[:300]}")

    def _record(self, role: str, thinking: bool, res: LLMResult, messages: list[dict[str, str]]) -> None:
        entry = {
            "n": len(self.calls) + 1,
            "role": role,
            "thinking": thinking,
            "ok": res.ok,
            "finish": res.finish_reason,
            "sec": round(res.seconds, 1),
            "usage": res.usage,
            "err": res.error[:200],
            "prompt_chars": sum(len(m["content"]) for m in messages),
            "content_chars": len(res.content),
            "reasoning_chars": len(res.reasoning),
        }
        self.calls.append(entry)
        log(f"house call {entry}")
        save_debug(
            f"call{entry['n']:02d}_{role}.txt",
            "\n\n".join(f"### {m['role']}\n{m['content']}" for m in messages)
            + f"\n\n### REASONING\n{res.reasoning}\n\n### CONTENT\n{res.content}\n\n### ERR\n{res.error}",
        )
