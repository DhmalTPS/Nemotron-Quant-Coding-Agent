# SPDX-License-Identifier: Apache-2.0
"""Deterministic verification of an attempt's outputs.

Errors are things a grader will certainly reject (missing file/key/column, wrong JSON type,
wrong row count, crash). Warnings are suspicious-but-possibly-legal findings, including
financial plausibility heuristics, that the reviewer model is asked to adjudicate.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

from .contract import Contract, FileSpec
from .runner import RunResult


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selfcheck: list[str] = field(default_factory=list)
    preview: str = ""
    files: list[str] = field(default_factory=list)

    @property
    def score(self) -> tuple[int, int, int]:
        """Lower is better. Used to keep the best attempt."""
        return (len(self.errors), len(self.selfcheck), len(self.warnings))

    @property
    def clean(self) -> bool:
        return not self.errors

    def summary(self, limit: int = 40) -> str:
        parts = []
        if self.errors:
            parts.append("ERRORS (must fix):\n" + "\n".join(f"  - {e}" for e in self.errors[:limit]))
        if self.selfcheck:
            parts.append("SELF-CHECK FAILURES (from the script's own checks):\n" + "\n".join(f"  - {e}" for e in self.selfcheck[:limit]))
        if self.warnings:
            parts.append("WARNINGS (check against the instruction; fix if they are real problems):\n" + "\n".join(f"  - {w}" for w in self.warnings[:limit]))
        return "\n".join(parts) if parts else "No problems detected by the automatic validator."


# ----------------------------------------------------------------------------- helpers
def _walk_json(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, k, v
            yield from _walk_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2000]):
            p = f"{path}[{i}]"
            yield from _walk_json(v, p)


def _get(obj: Any, dotted: str) -> tuple[bool, Any]:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            if not cur:
                return True, None
            cur = cur[0]
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


NAME_RULES: list[tuple[re.Pattern[str], str, Any]] = [
    (re.compile(r"(^|_)(prob|probability|survival|pd|hit_rate|win_rate|weight_sum)(_|$)", re.I), "unit", None),
    (re.compile(r"(^|_)(corr|correlation|rho|ic|rank_ic)(_|$)", re.I), "corr", None),
    (re.compile(r"(^|_)(price|premium|notional|vwap|spread_bps|variance)(_|$)", re.I), "nonneg", None),
    (re.compile(r"(^|_)(gamma|vega)(_|$)", re.I), "nonneg", None),
    (re.compile(r"(^|_)(sigma|vol|volatility|implied_vol|iv)(_|$)", re.I), "vol", None),
]


def _value_rules(name: str, vals: list[float]) -> list[str]:
    out = []
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    if not finite:
        return out
    lo, hi = min(finite), max(finite)
    for rx, kind, _ in NAME_RULES:
        if not rx.search(name):
            continue
        if kind == "unit" and (lo < -1e-9 or hi > 1 + 1e-9):
            out.append(f"'{name}' looks like a probability/fraction but ranges [{lo:.4g}, {hi:.4g}]")
        elif kind == "corr" and (lo < -1 - 1e-9 or hi > 1 + 1e-9):
            out.append(f"'{name}' looks like a correlation but ranges [{lo:.4g}, {hi:.4g}]")
        elif kind == "nonneg" and lo < -1e-9:
            out.append(f"'{name}' is negative somewhere (min {lo:.4g}); check sign conventions")
        elif kind == "vol" and hi > 5:
            out.append(f"'{name}' reaches {hi:.4g}; percent-vs-decimal confusion?")
        break
    return out


def _pairs_es_var(flat: dict[str, float]) -> list[str]:
    out = []
    for k, v in flat.items():
        m = re.search(r"(?<![a-z])(c?var)(?![a-z])", k, re.I)
        if not m or m.group(1).lower() != "var":
            continue
        for alt in ("es", "cvar", "expected_shortfall"):
            k2 = k[: m.start(1)] + alt + k[m.end(1):]
            if k2 in flat and flat[k2] is not None and v is not None:
                a, b = flat[k2], v
                if math.isfinite(a) and math.isfinite(b) and abs(a) + 1e-12 < abs(b) * (1 - 1e-6):
                    out.append(f"|{k2}|={a:.6g} < |{k}|={b:.6g}: expected shortfall should be >= VaR")
    return out


# ----------------------------------------------------------------------------- per file
def _check_json(p: pathlib.Path, spec: FileSpec | None, rep: Report) -> str:
    nan_paths: list[str] = []

    def bad_const(c: str) -> float:
        nan_paths.append(c)
        return float("nan")

    try:
        obj = json.loads(p.read_text(encoding="utf-8"), parse_constant=bad_const)
    except Exception as e:  # noqa: BLE001
        rep.errors.append(f"{p.name}: not valid JSON ({type(e).__name__}: {str(e)[:120]})")
        return ""
    if nan_paths:
        rep.warnings.append(f"{p.name}: contains NaN/Infinity literals ({len(nan_paths)}); graders usually reject NaN")
    if spec:
        for key, hint in spec.json_keys.items():
            found, val = _get(obj, key)
            if not found:
                rep.errors.append(f"{p.name}: missing key '{key}'")
                continue
            if val is None:
                rep.warnings.append(f"{p.name}: key '{key}' is null")
                continue
            if hint == "bool" and not isinstance(val, bool):
                rep.errors.append(f"{p.name}: '{key}' must be a JSON boolean, got {type(val).__name__} {str(val)[:30]}")
            elif hint == "int" and not (isinstance(val, int) and not isinstance(val, bool)):
                if isinstance(val, float) and val.is_integer():
                    rep.warnings.append(f"{p.name}: '{key}' should be an integer, got float {val}")
                else:
                    rep.errors.append(f"{p.name}: '{key}' must be an integer, got {type(val).__name__} {str(val)[:30]}")
            elif hint == "float" and (isinstance(val, (str, dict, list)) or isinstance(val, bool)):
                rep.errors.append(f"{p.name}: '{key}' must be a number, got {type(val).__name__}")
            elif hint == "object" and not isinstance(val, dict):
                rep.errors.append(f"{p.name}: '{key}' must be an object, got {type(val).__name__}")
            elif hint == "array" and not isinstance(val, list):
                rep.errors.append(f"{p.name}: '{key}' must be an array, got {type(val).__name__}")
            elif hint == "str" and not isinstance(val, str):
                rep.warnings.append(f"{p.name}: '{key}' expected a string, got {type(val).__name__}")
        # exact key sets are asserted by many graders: flag keys the template does not list
        template_parents: dict[str, set[str]] = {}
        for key in spec.json_keys:
            parent, _, leaf = key.rpartition(".")
            template_parents.setdefault(parent, set()).add(leaf)
        for parent, leaves in template_parents.items():
            if parent in spec.dynamic_parents or (parent == "" and "<root>" in spec.dynamic_parents):
                continue
            found, val = (True, obj) if parent == "" else _get(obj, parent)
            if found and isinstance(val, dict):
                extra = [k for k in val if k not in leaves]
                if extra:
                    rep.warnings.append(
                        f"{p.name}: keys not in the instruction's template under '{parent or '<root>'}': {extra[:8]} "
                        "(graders often assert the exact key set; remove unless the instruction asks for them)"
                    )
        for parent in spec.dynamic_parents:
            if parent == "<root>":
                continue
            found, val = _get(obj, parent)
            if not found:
                rep.errors.append(f"{p.name}: missing key '{parent}'")
            elif isinstance(val, (dict, list)) and not val:
                rep.errors.append(f"{p.name}: '{parent}' is empty")
    flat: dict[str, float] = {}
    for path, key, v in _walk_json(obj):
        n = _num(v)
        if n is not None:
            flat[path] = n
            if not math.isfinite(n):
                rep.warnings.append(f"{p.name}: '{path}' is not finite")
            for w in _value_rules(str(key), [n]):
                rep.warnings.append(f"{p.name}: {w}")
    for w in _pairs_es_var(flat):
        rep.warnings.append(f"{p.name}: {w}")
    # placeholder detection: a computation that silently fell through leaves its 0.0 defaults behind
    nums = [v for v in flat.values() if v is not None]
    zeros = sum(1 for v in nums if v == 0)
    if len(nums) >= 4 and zeros >= 0.5 * len(nums):
        rep.errors.append(
            f"{p.name}: {zeros} of {len(nums)} numeric values are exactly 0 - placeholder/default values left by "
            "computations that did not run. Remove default initialisations and try/except fallbacks; compute every value."
        )
    txt = json.dumps(obj, indent=1, default=str)
    return txt[:2600] + ("\n...(truncated)" if len(txt) > 2600 else "")


def _check_table(p: pathlib.Path, spec: FileSpec | None, rep: Report) -> str:
    import pandas as pd

    try:
        if p.suffix.lower() in (".parquet", ".pqt"):
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".xlsx":
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
    except Exception as e:  # noqa: BLE001
        rep.errors.append(f"{p.name}: cannot be read as a table ({type(e).__name__}: {str(e)[:120]})")
        return ""
    cols = [str(c) for c in df.columns]
    if any(c.startswith("Unnamed: 0") for c in cols):
        rep.errors.append(f"{p.name}: has an unnamed index column (write with index=False)")
    if df.empty:
        rep.errors.append(f"{p.name}: has no rows")
    if spec and spec.columns:
        missing = [c for c in spec.columns if c not in cols]
        if missing:
            rep.errors.append(f"{p.name}: missing columns {missing}; has {cols[:30]}")
        else:
            order = [c for c in cols if c in spec.columns]
            if order != list(spec.columns):
                rep.warnings.append(f"{p.name}: column order differs from the instruction {spec.columns}")
            extra = [c for c in cols if c not in spec.columns]
            if extra:
                rep.warnings.append(f"{p.name}: extra columns not in the instruction: {extra[:10]}")
    if spec and spec.n_rows is not None and len(df) != spec.n_rows:
        rep.errors.append(f"{p.name}: has {len(df)} rows, instruction says {spec.n_rows}")
    for c in df.columns:
        s = df[c]
        if len(s) and s.isna().all():
            rep.errors.append(f"{p.name}: column '{c}' is entirely empty/NaN")
        elif s.isna().any():
            rep.warnings.append(f"{p.name}: column '{c}' has {int(s.isna().sum())} NaN values")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = s.dropna().astype(float).tolist()
            for w in _value_rules(str(c), vals):
                rep.warnings.append(f"{p.name}: {w}")
            if vals and any(math.isinf(v) for v in vals):
                rep.warnings.append(f"{p.name}: column '{c}' has infinite values")
    try:
        with pd.option_context("display.width", 220, "display.max_columns", 40, "display.precision", 6):
            head = df.head(6).to_string(max_colwidth=28)
        desc = ""
        num = df.select_dtypes("number")
        if not num.empty:
            d = num.describe().loc[["min", "mean", "max"]].T
            desc = d.to_string(max_rows=40)[:1500]
        return f"shape={df.shape}\ncolumns={cols[:60]}\n{head[:2200]}\nnumeric summary:\n{desc}"
    except Exception:  # noqa: BLE001
        return f"shape={df.shape} columns={cols[:60]}"


def validate(stage: pathlib.Path, contract: Contract, run: RunResult | None, extra_expected: list[str]) -> Report:
    rep = Report()
    if run is not None:
        if run.timed_out:
            rep.errors.append(f"script timed out after {run.seconds:.0f}s (make it faster: vectorize, fewer paths/iterations if allowed)")
        elif run.returncode != 0:
            tb = run.stderr.strip().splitlines()
            last = tb[-1] if tb else ""
            rep.errors.append(f"script exited with code {run.returncode}: {last[:300]}")
        for line in (run.stdout + "\n" + run.stderr).splitlines():
            if "SELFCHECK_FAIL" in line:
                rep.selfcheck.append(line.split("SELFCHECK_FAIL", 1)[1].strip(" :")[:300])
    present = {}
    if stage.is_dir():
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                present[p.relative_to(stage).as_posix()] = p
    rep.files = list(present)
    expected: dict[str, bool] = {n: s.strong for n, s in contract.files.items()}
    for n in extra_expected:
        expected.setdefault(n, True)
    by_base = {pathlib.PurePosixPath(k).name: v for k, v in present.items()}
    previews = []
    for name, strong in expected.items():
        p = present.get(name) or by_base.get(pathlib.PurePosixPath(name).name)
        if p is None:
            (rep.errors if strong else rep.warnings).append(
                f"expected output file '{name}' was not written" + ("" if strong else " (it is mentioned in the instruction)")
            )
            continue
        if p.stat().st_size == 0:
            rep.errors.append(f"{name}: file is empty")
            continue
    if not present:
        rep.errors.append("no output files were written")
    for rel, p in present.items():
        spec = contract.files.get(rel) or contract.files.get(p.name)
        ext = p.suffix.lower()
        pv = ""
        if ext == ".json":
            pv = _check_json(p, spec, rep)
        elif ext in (".csv", ".tsv", ".parquet", ".pqt", ".xlsx"):
            pv = _check_table(p, spec, rep)
        elif ext == ".png":
            if p.read_bytes()[:4] != b"\x89PNG":
                rep.errors.append(f"{rel}: not a valid PNG")
            pv = f"(png, {p.stat().st_size:,} bytes)"
        else:
            try:
                pv = p.read_text(encoding="utf-8", errors="replace")[:600]
            except OSError:
                pv = ""
        if pv:
            previews.append(f"=== {rel} ===\n{pv}")
    unexpected = [r for r in present if r not in expected and pathlib.PurePosixPath(r).name not in expected]
    if unexpected and expected:
        rep.warnings.append(f"files written that the instruction does not mention: {unexpected[:10]}")
    text = "\n\n".join(previews)
    rep.preview = text[:9000] + ("\n...(preview truncated)" if len(text) > 9000 else "")
    return rep
