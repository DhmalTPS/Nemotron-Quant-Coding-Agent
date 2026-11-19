"""Python port of the starter pack's conformance sweep (Git Bash mangles docker -v paths).

For every public unit: run the image offline exactly as conformance.sh does (unit at /input:ro,
one dir at /output and /app/output), require exit 0, and require that the files the unit's
checker expects in the output directory exist. Expected names come from the checker sources
(dev/eval_contract.checker_files: quoted literals resolved against the output dir, excluding
reference/data/log roots).

    python dev/conformance.py [--image t1-agent:dev] [--jobs 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "track1-coding-public" / "units"
sys.path.insert(0, str(ROOT / "dev"))
from eval_contract import checker_files  # noqa: E402


def dp(p: pathlib.Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def one(image: str, unit: pathlib.Path, work: pathlib.Path) -> tuple[str, str]:
    out = work / unit.name
    out.mkdir(parents=True)
    cmd = ["docker", "run", "--rm", "--network=none", "-v", f"{dp(unit)}:/input:ro",
           "-v", f"{dp(out)}:/output", "-v", f"{dp(out)}:/app/output", image,
           "solve", "--task-dir", "/input", "--out", "/app/output"]
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600).returncode
    except subprocess.TimeoutExpired:
        return unit.name, "CRASH timeout"
    if rc != 0:
        return unit.name, f"CRASH exit {rc}"
    want = checker_files(unit)
    if not want:
        return unit.name, "UNCHECKED"
    have = {p.name for p in out.rglob("*") if p.is_file()}
    missing = sorted(want - have)
    if len(missing) == len(want):
        return unit.name, f"WROTE_NOTHING_EXPECTED {missing[:6]}"
    return unit.name, "ok" if not missing else f"ok(partial; missing {missing[:6]})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="t1-agent:dev")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    work = ROOT / "benchmarks" / "conformance_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    units = sorted(p for p in KIT.iterdir() if p.is_dir())
    res = []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, status in ex.map(lambda u: one(a.image, u, work), units):
            res.append((name, status))
            if status != "ok":
                print(f"{name:46s} {status}", flush=True)
    c = {k: sum(1 for _, s in res if s.startswith(k)) for k in ("ok", "CRASH", "WROTE_NOTHING", "UNCHECKED")}
    print(f"units {len(res)}   ok {c['ok']}   crashed {c['CRASH']}   wrote-nothing-expected {c['WROTE_NOTHING']}   unchecked {c['UNCHECKED']}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
