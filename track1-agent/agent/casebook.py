# SPDX-License-Identifier: Apache-2.0
"""Offline casebook of our own solutions to OTHER public units (competition rule 8).

Rule 8: our solutions to public Development units may be used as in-context examples when
solving *other* units; never on the unit they solve (no stored answer, however the unit is
recognised -- by id or by content). Enforcement here is two-layered:
  1. exclude any case whose unit_id equals the target's id;
  2. exclude any case whose instruction is textually similar to the target's (content match),
     so a renamed copy of a public unit can never retrieve its own solution.
Cases contain only code we wrote/generated and short notes; canary GUIDs are stripped at build
time and the final output scan removes any that might slip through.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter

from .task import Task

CASEBOOK_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets" / "casebook"
MAX_CHARS = 7000
SIM_EXCLUDE = 0.90  # >= this: treat as the same unit (renamed copy) and exclude. Self-similarity is 1.0;
#                      different-but-related public units measure <= 0.75.
SIM_MIN = 0.30      # below this a case is not relevant enough to be worth the prompt space

_STOP = set("the a an of to and or in on for with by is are be as at from that this it its into use using each "
            "all any must should will can not no than then per your you file files output outputs input data "
            "column columns value values json csv compute calculate return returns".split())


def _tokens(text: str) -> Counter[str]:
    toks = re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
    return Counter(t for t in toks if t not in _STOP)


def _cos(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def _load() -> list[dict]:
    idx = CASEBOOK_DIR / "index.json"
    if not idx.exists():
        return []
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def retrieve(task: Task, k: int = 1) -> str:
    cases = _load()
    if not cases:
        return ""
    target = _tokens(task.instruction)
    scored = []
    for c in cases:
        if c.get("unit_id") == task.unit_id:
            continue
        ctoks = Counter(c.get("tokens", {}))
        sim = _cos(target, ctoks)
        if sim >= SIM_EXCLUDE:
            continue  # content match: could be the same unit under another name
        bonus = 0.05 if c.get("category") == task.category else 0.0
        scored.append((sim + bonus, c))
    scored.sort(key=lambda x: -x[0])
    out = []
    used = 0
    for s, c in scored[:k]:
        if s < SIM_MIN:
            break
        p = CASEBOOK_DIR / c["file"]
        try:
            code = p.read_text(encoding="utf-8")
        except OSError:
            continue
        block = f"--- example: {c.get('title', '')} (category {c.get('category', '')}) ---\n"
        if c.get("notes"):
            block += "notes: " + c["notes"] + "\n"
        block += code[: MAX_CHARS - len(block)]
        if used + len(block) > MAX_CHARS * k:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)
