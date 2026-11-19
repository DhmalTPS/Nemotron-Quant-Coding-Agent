"""Measure deliverable-filename extraction (instruction only) against the checkers' filenames.

Usage: python dev/eval_contract.py <units_dir> [unit_substring]
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.contract import OUT_EXT, extract  # noqa: E402
from agent.task import load_task  # noqa: E402

EXT = "|".join(OUT_EXT)


def checker_files(unit: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for p in (unit / "checks").glob("*.py"):
        t = p.read_text(encoding="utf8", errors="replace")
        for m in re.finditer(r"""['"]([\w./-]+\.(?:%s))['"]""" % EXT, t):
            nm = m.group(1)
            if nm.startswith(("/app/data", "/tests", "/logs", "/app/tests")) or "reference" in nm:
                continue
            if nm.startswith("/") and not nm.startswith(("/app/output", "/output")):
                continue
            base = nm.split("/")[-1]
            if base in ("reward.json", "pytest_report.json", "reward.txt", "diagnostic.json", "reward_summary.json"):
                continue
            names.add(base)
    shipped = {p.name for p in unit.rglob("*") if p.is_file()}  # inputs AND checker reference files
    return names - shipped


def main() -> None:
    units = pathlib.Path(sys.argv[1])
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    tot_tp = tot_fn = tot_fp = 0
    for u in sorted(units.iterdir()):
        if flt and flt not in u.name:
            continue
        task = load_task(str(u), "/tmp/out", use_checks=False)
        c = extract(task)
        got = set(c.strong_files)
        weak = set(c.files) - got
        truth = checker_files(u)
        tp, fn, fp = got & truth, truth - got, got - truth
        tot_tp += len(tp)
        tot_fn += len(fn)
        tot_fp += len(fp)
        flag = "OK " if not fn and not fp else "!! "
        if flt or fn or fp:
            print(f"{flag}{u.name[:42]:42s} miss={sorted(fn)} extra={sorted(fp)} weak={sorted(weak)[:6]}")
        if flt:
            print(c.describe())
    print(f"TP={tot_tp} FN={tot_fn} FP={tot_fp}")


if __name__ == "__main__":
    main()
