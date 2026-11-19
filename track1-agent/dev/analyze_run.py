"""Failure taxonomy for a benchmark run.

    python dev/analyze_run.py benchmarks/runs/<tag>

Buckets each unit:
  PASS            reward 1
  SILENT_WRONG    agent's own validator was clean but the grader failed it (semantics/conventions)
  KNOWN_FAIL      agent ended with validator errors (crash / schema / missing file)
  NO_CODE         no attempt ran (model/plumbing failure)
and prints time / request statistics so budget decisions are data-driven.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import statistics
import sys


def main() -> None:
    run = pathlib.Path(sys.argv[1])
    rows = list(csv.DictReader(open(run / "results.csv", encoding="utf-8")))
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    times, reqs = [], []
    lines = []
    for r in sorted(rows, key=lambda r: r["unit"]):
        uid = r["unit"]
        summ = {}
        p = run / uid / "debug" / "summary.json"
        if p.exists():
            summ = json.loads(p.read_text(encoding="utf-8"))
        atts = summ.get("attempts", [])
        best = next((a for a in atts if a["n"] == summ.get("best")), None)
        reward = float(r.get("reward") or 0)
        if reward >= 1:
            b = "PASS"
        elif not atts:
            b = "NO_CODE"
        elif best and best["score"][0] == 0:
            b = "SILENT_WRONG"
        else:
            b = "KNOWN_FAIL"
        buckets[b].append(uid)
        times.append(float(r.get("agent_s") or 0))
        reqs.append(int(float(r.get("requests") or 0)))
        origins = ",".join(a["origin"] for a in atts)
        err = (best or {}).get("errors", [""])[:1]
        lines.append(f"{b:12s} {uid[:40]:40s} t={r.get('agent_s')}s req={r.get('requests')} att=[{origins}] "
                     f"fail={r.get('failing', '')[:80]} {('| ' + err[0][:80]) if err and err[0] else ''}")
    print("\n".join(lines))
    n = len(rows)
    print("\n== buckets ==")
    for k in ("PASS", "SILENT_WRONG", "KNOWN_FAIL", "NO_CODE"):
        print(f"{k:12s} {len(buckets[k]):3d}  {len(buckets[k]) / max(n, 1):.1%}")
    if times:
        print(f"\ntime: mean {statistics.mean(times):.0f}s  p90 {sorted(times)[int(0.9 * (len(times) - 1))]:.0f}s  max {max(times):.0f}s"
              f"  | 86-unit projection {statistics.mean(times) * 86 / 3600:.1f} h (stage clock 12 h)")
        print(f"requests: mean {statistics.mean(reqs):.1f}  max {max(reqs)}")


if __name__ == "__main__":
    main()
