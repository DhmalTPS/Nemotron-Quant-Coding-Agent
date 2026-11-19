# SPDX-License-Identifier: Apache-2.0
"""Deterministic data profiling so the model never guesses a column name, dtype, unit scale
or file format. Bounded output: the profile is a compact text block, never raw data dumps."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from typing import Any

from .task import InputFile, Task, scrub

PER_FILE = 2600
TOTAL = 22000
TEXT_EXT = {".md", ".txt", ".py", ".xml", ".html", ".htm", ".yaml", ".yml", ".toml", ".sql", ".r", ".ini", ".cfg"}


def _fmt_num(x: Any) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:
        return "nan"
    a = abs(xf)
    if a != 0 and (a < 1e-3 or a >= 1e7):
        return f"{xf:.4e}"
    return f"{xf:.6g}"


def _quiet_dates(s: Any) -> Any:
    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def _profile_frame(df: Any, name: str, nrows_total: int | None = None) -> str:
    import pandas as pd

    lines = [f"shape={nrows_total if nrows_total is not None else len(df)} rows x {df.shape[1]} cols"]
    cols = list(df.columns)
    show_cols = cols[:60]
    for c in show_cols:
        s = df[c]
        nnull = int(s.isna().sum())
        desc = f"  - {c!r}: {s.dtype}"
        if nnull:
            desc += f", nulls={nnull}"
        if pd.api.types.is_bool_dtype(s):
            desc += f", true={int(s.sum())}"
        elif pd.api.types.is_numeric_dtype(s):
            sv = s.dropna()
            if len(sv):
                desc += (
                    f", min={_fmt_num(sv.min())}, max={_fmt_num(sv.max())}, mean={_fmt_num(sv.mean())}"
                    f", nunique={sv.nunique()}"
                )
        else:
            sv = s.dropna().astype(str)
            nu = sv.nunique()
            desc += f", nunique={nu}"
            if nu <= 12:
                desc += f", values={sorted(sv.unique().tolist())[:12]}"
            else:
                ex = sv.iloc[:3].tolist()
                desc += f", e.g.={[e[:40] for e in ex]}"
                # date-like?
                try:
                    d = _quiet_dates(sv.iloc[: min(len(sv), 2000)])
                    if d.notna().mean() > 0.9:
                        dd = _quiet_dates(sv)
                        desc += f", parses_as_dates [{dd.min()} .. {dd.max()}], sorted={dd.is_monotonic_increasing}"
                except Exception:  # noqa: BLE001
                    pass
        lines.append(desc)
    if len(cols) > len(show_cols):
        lines.append(f"  ... {len(cols) - len(show_cols)} more columns: {cols[60:120]}")
    try:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            head = df.head(4).to_csv(index=False)
            lines.append("first rows (csv):\n" + "\n".join(head.splitlines()[:5])[:900])
            if len(df) > 4:
                tail = df.tail(2).to_csv(index=False, header=False)
                lines.append("last rows:\n" + tail[:400])
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def _json_shape(obj: Any, depth: int = 0, max_depth: int = 4) -> str:
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"{pad}{{...{len(obj)} keys}}"
        out = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= 25:
                out.append(f"{pad}... {len(obj) - 25} more keys")
                break
            if isinstance(v, (dict, list)):
                out.append(f"{pad}{k!r}: {type(v).__name__}(len={len(v)})")
                out.append(_json_shape(v, depth + 1, max_depth))
            else:
                out.append(f"{pad}{k!r}: {json.dumps(v)[:80]}")
        return "\n".join(out)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]"
        first = obj[0]
        if isinstance(first, (dict, list)):
            return f"{pad}[{len(obj)} x {type(first).__name__}], first element:\n" + _json_shape(first, depth + 1, max_depth)
        return f"{pad}[{len(obj)} items] e.g. {json.dumps(obj[:8])[:200]}"
    return f"{pad}{json.dumps(obj)[:120]}"


def profile_file(f: InputFile) -> str:
    import pandas as pd

    ext = f.path.suffix.lower()
    head = f"### {f.path}  ({f.size:,} bytes)"
    if f.image_paths:
        head += f"\n(instruction may call it: {', '.join(f.image_paths[:3])})"
    try:
        is_table = ext in (".csv", ".tsv") or (ext == ".txt" and f.size < 5_000_000 and _looks_tabular(f))
        if is_table:
            sep = "\t" if ext == ".tsv" else None
            nrows = None if f.size < 40_000_000 else 300_000
            df = pd.read_csv(f.path, sep=sep, engine="python" if sep is None else "c", nrows=nrows)
            total = None
            if nrows is not None:
                with open(f.path, "rb") as fh:
                    total = sum(1 for _ in fh) - 1
            raw_head = _raw_head(f, 3)
            return head + f"\nraw first lines:\n{raw_head}\n" + _profile_frame(df, f.rel, total)
        if ext in (".parquet", ".pqt"):
            df = pd.read_parquet(f.path)
            return head + "\n" + _profile_frame(df, f.rel)
        if ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(f.path)
            parts = [head, f"sheets={xl.sheet_names}"]
            for sh in xl.sheet_names[:4]:
                df = xl.parse(sh)
                parts.append(f"-- sheet {sh!r}\n" + _profile_frame(df, sh))
            return "\n".join(parts)
        if ext == ".json":
            text = f.path.read_text(encoding="utf-8", errors="replace")
            if len(text) <= 1800:
                return head + "\nfull content:\n" + scrub(text)
            obj = json.loads(text)
            return head + "\nstructure:\n" + _json_shape(obj)
        if ext == ".jsonl":
            lines = f.path.read_text(encoding="utf-8", errors="replace").splitlines()
            return head + f"\n{len(lines)} lines; first 2:\n" + "\n".join(l[:600] for l in lines[:2])
        if ext == ".zip":
            with zipfile.ZipFile(f.path) as z:
                names = z.infolist()
                parts = [head, f"zip with {len(names)} members:"]
                for zi in names[:30]:
                    parts.append(f"  {zi.filename} ({zi.file_size:,} bytes)")
                for zi in names[:2]:
                    if zi.filename.lower().endswith((".csv", ".txt", ".json", ".tsv")) and zi.file_size:
                        with z.open(zi) as fh:
                            snippet = fh.read(600).decode("utf-8", "replace")
                        parts.append(f"  first bytes of {zi.filename}:\n{snippet}")
                return "\n".join(parts)
        if ext in TEXT_EXT or f.size < 200_000:
            text = scrub(f.path.read_text(encoding="utf-8", errors="replace"))
            n = len(text)
            return head + f"\ntext, {n:,} chars; beginning:\n" + text[:1500] + ("\n..." if n > 1500 else "")
        return head + "\n(binary; not profiled)"
    except Exception as e:  # noqa: BLE001
        return head + f"\n(profile failed: {type(e).__name__}: {str(e)[:200]})\nraw first lines:\n{_raw_head(f, 4)}"


def _raw_head(f: InputFile, n: int) -> str:
    try:
        with open(f.path, encoding="utf-8", errors="replace") as fh:
            return "".join(next(fh, "") for _ in range(n))[:800]
    except OSError:
        return ""


def _looks_tabular(f: InputFile) -> bool:
    h = _raw_head(f, 3).splitlines()
    return len(h) >= 2 and all(l.count(",") >= 2 for l in h)


def build_profile(task: Task) -> str:
    """Profile every input. Files with identical headers are grouped to save tokens."""
    groups: dict[str, list[InputFile]] = defaultdict(list)
    for f in task.files:
        key = f.rel
        if f.path.suffix.lower() in (".csv", ".tsv"):
            key = "hdr:" + _raw_head(f, 1).strip()
        groups[key].append(f)
    parts: list[str] = []
    used = 0
    for _key, fs in groups.items():
        fs = sorted(fs, key=lambda x: x.rel)
        block = profile_file(fs[0])[:PER_FILE]
        if len(fs) > 1:
            others = ", ".join(str(x.path.name) for x in fs[1:])
            block += f"\n(+ {len(fs) - 1} more files with the SAME header: {others[:600]})"
        if used + len(block) > TOTAL:
            parts.append(f"### {fs[0].path} ... (profile budget exhausted; {len(fs)} file(s), inspect in code)")
            continue
        parts.append(block)
        used += len(block)
    if not parts:
        parts.append("(no input data files: everything needed is in the instruction)")
    return "\n\n".join(parts)


def inventory(task: Task) -> str:
    lines = []
    for f in task.files:
        alias = f"   <- instruction alias: {', '.join(f.image_paths[:3])}" if f.image_paths else ""
        lines.append(f"{f.path}  ({f.size:,} B){alias}")
    return "\n".join(lines) if lines else "(none)"
