"""Run ingestion + profiling + contract on every unit; report time, prompt size and failures."""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import prompts  # noqa: E402
from agent.contract import extract  # noqa: E402
from agent.profile import build_profile, inventory  # noqa: E402
from agent.task import card_summary, load_task  # noqa: E402


def main() -> None:
    units = pathlib.Path(sys.argv[1])
    worst = []
    for u in sorted(units.iterdir()):
        t0 = time.time()
        try:
            task = load_task(str(u), "/tmp/out", use_checks=False)
            prof = build_profile(task)
            c = extract(task)
            ctx = prompts.task_block(task.instruction, card_summary(task), inventory(task), prof, c.describe(), "", "")
            dt = time.time() - t0
            fails = prof.count("profile failed")
            canary = any(g in ctx.lower() for g in task.canaries)
            worst.append((dt, u.name, len(ctx), fails, canary))
        except Exception as e:  # noqa: BLE001
            worst.append((time.time() - t0, u.name, -1, 99, False))
            print(f"CRASH {u.name}: {type(e).__name__}: {e}")
    worst.sort(reverse=True)
    for dt, name, n, fails, canary in worst[:12]:
        print(f"{dt:6.2f}s ctx={n:6d} chars profile_fail={fails} canary_in_prompt={canary} {name}")
    print("max ctx chars", max(w[2] for w in worst), "| units with canary leak:", sum(w[4] for w in worst),
          "| profile failures:", sum(w[3] for w in worst if w[3] != 99))


if __name__ == "__main__":
    main()
