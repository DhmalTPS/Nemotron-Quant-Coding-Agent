# SPDX-License-Identifier: Apache-2.0
"""Deterministic, semantics-preserving hardening of generated code.

stable_sorts: pandas sort_values/sort_index and numpy argsort default to quicksort, which is NOT
stable. When ties exist (duplicate dates, equal scores) the surviving order is arbitrary, so
`sort -> drop_duplicates(keep="last")` or "break ties by original order" silently differs from the
oracle. Verified on t1-cross-sectional-momentum: datetime + quicksort keeps the wrong duplicate,
stable sort reproduces the oracle exactly. Every call without an explicit `kind=` gets
kind="stable"; the edit is made in place (no reformatting) and abandoned if the result does not
compile.
"""

from __future__ import annotations

import ast

SORT_ATTRS = {"sort_values", "sort_index", "argsort"}


def stable_sorts(code: str) -> tuple[str, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    lines = code.split("\n")
    edits: list[tuple[int, int, bool]] = []  # (line index, col of ')', has_args)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in SORT_ATTRS:
            continue
        if any(k.arg == "kind" for k in node.keywords) or any(k.arg is None for k in node.keywords):
            continue  # explicit kind, or **kwargs we cannot see into
        if node.end_lineno is None or node.end_col_offset is None:
            continue
        li, col = node.end_lineno - 1, node.end_col_offset - 1
        if li >= len(lines) or col >= len(lines[li]) or lines[li][col] != ")":
            continue
        edits.append((li, col, bool(node.args or node.keywords)))
    if not edits:
        return code, 0
    for li, col, has_args in sorted(edits, reverse=True):
        line = lines[li]
        # last significant character before ')' may be on an earlier line (multi-line calls)
        prefix = ("\n".join(lines[:li]) + "\n" + line[:col]).rstrip()
        if not has_args or prefix.endswith("("):
            ins = 'kind="stable"'
        elif prefix.endswith(","):
            ins = 'kind="stable"'
        else:
            ins = ', kind="stable"'
        lines[li] = line[:col] + ins + line[col:]
    new = "\n".join(lines)
    try:
        compile(new, "solution.py", "exec")
    except SyntaxError:
        return code, 0
    return new, len(edits)
