# SPDX-License-Identifier: Apache-2.0
"""Code extraction, patch application, path remapping and static safety checks."""

from __future__ import annotations

import ast
import re

from .task import Task

FENCE_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)(?:```|\Z)", re.S | re.I)
PATCH_RE = re.compile(
    r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n]*\n(.*?)\n?>{5,9} ?REPLACE", re.S
)

PRELUDE_MARK = "# ==== AGENT PRELUDE (auto-generated, do not edit) ===="
PRELUDE_END = "# ==== END AGENT PRELUDE ===="


def extract_code(text: str) -> str:
    """Return the largest python code block (or the whole text if it parses as python)."""
    blocks = [b for b in FENCE_RE.findall(text) if b.strip()]
    if blocks:
        return max(blocks, key=len).rstrip() + "\n"
    stripped = text.strip()
    if stripped:
        try:
            ast.parse(stripped)
            return stripped + "\n"
        except SyntaxError:
            pass
    return ""


def strip_fences(text: str) -> str:
    t = re.sub(r"^\s*```[\w]*\s*\n", "", text)
    t = re.sub(r"\n```\s*$", "\n", t)
    return t


def join_continuation(first: str, cont: str) -> str:
    """Join a truncated code reply with its continuation, removing duplicated overlap."""
    a = first
    m = FENCE_RE.search(a)
    if m:
        a = a[m.start(1):]
    a = a.rstrip("`")
    b = strip_fences(cont)
    # drop overlapping prefix of b that repeats the tail of a
    tail = a[-400:]
    for k in range(min(len(tail), len(b)), 20, -1):
        if tail.endswith(b[:k]):
            b = b[k:]
            break
    # if continuation restarts the last (partial) line, drop the partial line from a
    last_nl = a.rfind("\n")
    partial = a[last_nl + 1 :]
    if partial and b.lstrip().startswith(partial.strip()[:30]) and len(partial.strip()) > 5:
        a = a[: last_nl + 1]
    return (a + b).replace("```", "")


def syntax_error(code: str) -> str:
    try:
        ast.parse(code)
        return ""
    except SyntaxError as e:
        line = (e.text or "").rstrip()
        return f"SyntaxError: {e.msg} at line {e.lineno}: {line[:200]}"


def apply_patches(code: str, reply: str) -> tuple[str, int, list[str]]:
    """Apply SEARCH/REPLACE blocks. Returns (new_code, applied, failures)."""
    applied = 0
    failures: list[str] = []
    for search, replace in PATCH_RE.findall(reply):
        if not search.strip():
            failures.append("empty SEARCH block")
            continue
        if search in code:
            code = code.replace(search, replace, 1)
            applied += 1
            continue
        # whitespace-tolerant line matching
        new = _fuzzy_replace(code, search, replace)
        if new is not None:
            code = new
            applied += 1
        else:
            failures.append(search.strip().splitlines()[0][:120])
    return code, applied, failures


def _fuzzy_replace(code: str, search: str, replace: str) -> str | None:
    lines = code.split("\n")
    s_lines = [l.strip() for l in search.strip("\n").split("\n")]
    s_lines = [l for l in s_lines if l]
    if not s_lines:
        return None
    n = len(s_lines)
    stripped = [l.strip() for l in lines]
    idx_nonempty = [i for i, l in enumerate(stripped) if l]
    for start in range(len(idx_nonempty)):
        if stripped[idx_nonempty[start]] != s_lines[0]:
            continue
        if start + n > len(idx_nonempty):
            break
        if all(stripped[idx_nonempty[start + k]] == s_lines[k] for k in range(n)):
            a = idx_nonempty[start]
            b = idx_nonempty[start + n - 1]
            indent_src = re.match(r"\s*", lines[a]).group(0)
            rep_lines = replace.strip("\n").split("\n")
            first_indent = re.match(r"\s*", rep_lines[0]).group(0) if rep_lines else ""
            fixed = []
            for rl in rep_lines:
                if rl.startswith(first_indent):
                    fixed.append(indent_src + rl[len(first_indent):])
                else:
                    fixed.append(rl)
            return "\n".join(lines[:a] + fixed + lines[b + 1 :])
    return None


# --------------------------------------------------------------------------- path handling
def remap_paths(code: str, task: Task, stage_dir: str) -> str:
    """Rewrite unit-image input paths to their real location and output paths to the stage
    directory. Applied to string literals only, longest path first."""
    mapping: dict[str, str] = {}
    for img, real in task.path_map.items():
        if img in ("/input",) or img.startswith(("/app/output", "/output")):
            continue
        mapping[img] = real.replace("\\", "/")  # forward slashes: safe inside any string literal
    mapping["/app/output"] = stage_dir.replace("\\", "/")
    mapping["/output"] = stage_dir.replace("\\", "/")
    keys = sorted(mapping, key=len, reverse=True)

    def fix_literal(m: re.Match[str]) -> str:
        q, body = m.group(1), m.group(2)
        for k in keys:
            if body == k or body.startswith(k + "/") or body.startswith(k + "\\"):
                body = mapping[k] + body[len(k):]
                break
        return q + body + q

    # simple single/double quoted literals (incl. f-string bodies, which are just text)
    return re.sub(r"""(['"])(/(?:app|output|tests|data)[^'"\n]*)\1""", fix_literal, code)


_PRELUDE_HELPERS = '''
def to_builtin(obj):
    """Recursively convert numpy/pandas scalars and containers to JSON-native Python types."""
    import math as _m
    try:
        import numpy as _np
    except Exception:  # pragma: no cover
        _np = None
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if _np is not None:
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return [to_builtin(v) for v in obj.tolist()]
    if hasattr(obj, "isoformat") and not isinstance(obj, str):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def write_json(name, obj, indent=2):
    """Write obj to OUTPUT_DIR/name (or an absolute path) with JSON-native types."""
    import json as _json
    path = name if _os.path.isabs(str(name)) else _os.path.join(OUTPUT_DIR, str(name))
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as _fh:
        _json.dump(to_builtin(obj), _fh, indent=indent)
    return path
'''


def build_prelude(task: Task, stage_dir: str) -> str:
    return (
        f"{PRELUDE_MARK}\n"
        "import os as _os\n"
        f"TASK_DIR = {str(task.task_dir)!r}\n"
        f"DATA_DIR = {str(task.data_dir)!r}\n"
        f"OUTPUT_DIR = {stage_dir!r}\n"
        "_os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
        "_os.environ['OUTPUT_DIR'] = OUTPUT_DIR\n"
        f"{_PRELUDE_HELPERS}\n"
        f"{PRELUDE_END}\n"
    )


def strip_prelude(code: str) -> str:
    if PRELUDE_MARK in code and PRELUDE_END in code:
        a = code.index(PRELUDE_MARK)
        b = code.index(PRELUDE_END) + len(PRELUDE_END)
        return (code[:a] + code[b:]).lstrip("\n")
    return code


# --------------------------------------------------------------------------- safety
BANNED = [
    (re.compile(r"\bpip\s+install|subprocess\.[a-z_]+\([^)]*pip"), "runtime package installation (no network)"),
    (re.compile(r"\b(requests|httpx|urllib\.request|urllib3|aiohttp)\b.*\b(get|post|urlopen)\b"), "network access"),
    (re.compile(r"\byfinance\b|\bpandas_datareader\b"), "external data download"),
    (re.compile(r"MODEL_TOKEN|MODEL_ENDPOINT"), "model credentials inside solution code"),
]


def static_issues(code: str) -> list[str]:
    issues = []
    for rx, why in BANNED:
        if rx.search(code):
            issues.append(f"forbidden: {why}")
    if re.search(r"""['"]\.?/?output/""", code) and "OUTPUT_DIR" not in code:
        issues.append("writes to a relative output path; use OUTPUT_DIR")
    return issues


def numbered(code: str, max_lines: int = 900) -> str:
    lines = code.split("\n")
    out = [f"{i + 1:4d}| {l}" for i, l in enumerate(lines[:max_lines])]
    if len(lines) > max_lines:
        out.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(out)
