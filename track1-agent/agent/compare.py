# SPDX-License-Identifier: Apache-2.0
"""Differential comparison of two attempts' deliverables.

Two independent implementations of the same specification that agree on every number are very
likely right (interpretation errors are rarely made identically twice); where they disagree, the
exact disagreements are what an adjudicator needs to see.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

REL = 1e-6
ABS = 1e-9


def _close(a: float, b: float, rel: float = REL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(ABS, rel * max(abs(a), abs(b)))


def _flat(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _cmp_json(pa: pathlib.Path, pb: pathlib.Path, name: str, diffs: list[str]) -> None:
    try:
        a = _flat(json.loads(pa.read_text(encoding="utf-8")))
        b = _flat(json.loads(pb.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diffs.append(f"{name}: key '{k}' only in {'A' if k in a else 'B'}")
            continue
        va, vb = a[k], b[k]
        if isinstance(va, bool) or isinstance(vb, bool) or not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            if va != vb:
                diffs.append(f"{name}: '{k}' A={str(va)[:60]!r} B={str(vb)[:60]!r}")
        elif not _close(float(va), float(vb)):
            diffs.append(f"{name}: '{k}' A={va:.10g} B={vb:.10g}")


def _read_table(p: pathlib.Path) -> Any:
    import pandas as pd

    if p.suffix.lower() in (".parquet", ".pqt"):
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")


def _cmp_table(pa: pathlib.Path, pb: pathlib.Path, name: str, diffs: list[str]) -> None:
    import pandas as pd

    try:
        a, b = _read_table(pa), _read_table(pb)
    except Exception:  # noqa: BLE001
        return
    if list(a.columns) != list(b.columns):
        diffs.append(f"{name}: columns differ A={list(a.columns)[:12]} B={list(b.columns)[:12]}")
    if len(a) != len(b):
        diffs.append(f"{name}: row count A={len(a)} B={len(b)}")
        return
    for c in [c for c in a.columns if c in b.columns]:
        sa, sb = a[c], b[c]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb) and not pd.api.types.is_bool_dtype(sa):
            x, y = sa.astype(float).to_numpy(), sb.astype(float).to_numpy()
            bad = [i for i in range(len(x)) if not _close(x[i], y[i])]
            if bad:
                i = bad[0]
                diffs.append(f"{name}: column '{c}' differs in {len(bad)}/{len(x)} rows (first row {i}: A={x[i]:.10g} B={y[i]:.10g})")
        else:
            ne = (sa.astype(str) != sb.astype(str)).to_numpy()
            if ne.any():
                i = int(ne.argmax())
                diffs.append(f"{name}: column '{c}' differs in {int(ne.sum())} rows (first row {i}: A={str(sa.iloc[i])[:40]!r} B={str(sb.iloc[i])[:40]!r})")


def compare_dirs(da: pathlib.Path, db: pathlib.Path, names: list[str]) -> list[str]:
    diffs: list[str] = []
    for name in names:
        pa, pb = da / name, db / name
        if not pa.exists() or not pb.exists():
            if pa.exists() != pb.exists():
                diffs.append(f"{name}: written only by {'A' if pa.exists() else 'B'}")
            continue
        ext = pa.suffix.lower()
        if ext == ".json":
            _cmp_json(pa, pb, name, diffs)
        elif ext in (".csv", ".tsv", ".parquet", ".pqt"):
            _cmp_table(pa, pb, name, diffs)
    return diffs
