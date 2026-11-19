"""Local benchmark: run the agent image on public units exactly the way the platform does, then
grade each output with the unit's own checker in the official sandbox image.

Platform fidelity: --read-only, uid 65534, cap-drop ALL, no-new-privileges, 64 MiB noexec /tmp,
256 PIDs, 1024 fds, unit mounted read-only at /input, one output dir bound at /output AND
/app/output. By default checks/ is REMOVED from the agent's view (it is sealed on the platform),
so the score estimates hidden-unit performance.

Model access for local runs: an OpenAI-compatible endpoint configured via files in
extra_info/keys (never printed):  dev_llm_key.txt (+ optional dev_llm_endpoint.txt,
dev_llm_model.txt). Defaults: NVIDIA API, nvidia/nemotron-3-super-120b-a12b.

Usage:
  python dev/bench.py --image t1-agent:dev --units panel   [--show-checks] [--jobs 2] [--tag name]
  python dev/bench.py --image t1-agent:dev --units t1-sma-crossover-spy,t1-var-es-estimation
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "track1-coding-public"
KEYS = ROOT.parent / "extra_info" / "keys"
RUNS = ROOT / "benchmarks" / "runs"
MEM = os.environ.get("BENCH_MEM", "3500m")  # per container; host VM has ~7.9 GB
sys.path.insert(0, str(ROOT))
from agent.task import build_path_map  # noqa: E402

PANEL = [
    "t1-sma-crossover-spy", "t1-fx-forward-cross-rate", "t1-historical-var-data-prep",
    "t1-earnings-surprise-calculator", "t1-var-es-estimation", "t1-implied-vol-approximations",
    "t1-fft-compound-poisson", "t1-alpha-hedge-strategy", "t1-regime-riskparity-cvar",
    "t1-crypto-funding-rate-basis-carry", "t1-swap-curve-bootstrap-ois", "t1-hull-white-swaption",
    "t1-cross-sectional-momentum", "t1-credit-migration-matrix", "t1-etf-cross-asset-lead-lag",
    "t1-EXAMPLE-bs-greeks-pde",
]


def _key(name: str) -> str:
    p = KEYS / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def model_env() -> dict[str, str]:
    token = _key("dev_llm_key.txt")
    if not token:
        return {}
    return {
        "MODEL_ENDPOINT": _key("dev_llm_endpoint.txt") or "https://integrate.api.nvidia.com",
        "MODEL_NAME": _key("dev_llm_model.txt") or "nvidia/nemotron-3-super-120b-a12b",
        "MODEL_TOKEN": token,
    }


def dpath(p: pathlib.Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def card_resources(unit: pathlib.Path) -> tuple[int, float]:
    import tomllib

    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    cpus = int(card.get("environment", {}).get("cpus", 4))
    timeout = float(card.get("agent", {}).get("timeout_sec", 1800))
    return cpus, timeout


def run_unit(image: str, unit: pathlib.Path, run_dir: pathlib.Path, show_checks: bool, extra_env: dict[str, str]) -> dict:
    uid = unit.name
    base = run_dir / uid
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    inp, out, dbg = base / "input", base / "output", base / "debug"
    shutil.copytree(unit, inp, ignore=None if show_checks else shutil.ignore_patterns("checks"))
    out.mkdir(parents=True)
    dbg.mkdir(parents=True)
    for d in (out, dbg):
        os.chmod(d, 0o777)
    cpus, timeout = card_resources(unit)
    host_cpus = os.cpu_count() or 4
    env = dict(model_env())
    env.update(extra_env)
    cmd = [
        "docker", "run", "--rm", "--read-only", "--user", "65534:65534", "--cap-drop=ALL",
        "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--pids-limit", "256", "--ulimit", "nofile=1024:1024", "--ulimit", "nproc=256:256",
        "--cpus", str(min(cpus, host_cpus)), "--memory", MEM, "--memory-swap", MEM,
        "-v", f"{dpath(inp)}:/input:ro", "-v", f"{dpath(out)}:/output", "-v", f"{dpath(out)}:/app/output",
        "-v", f"{dpath(dbg)}:/debug", "-e", "AGENT_DEBUG_DIR=/debug", "-e", "QFBENCH_SEED=0",
        "-e", "QFBENCH_NETWORK=restricted",
    ]
    if not env.get("MODEL_TOKEN"):
        cmd += ["--network=none"]
    for k, v in env.items():
        os.environ[k] = v
        cmd += ["-e", k]  # value passed through the environment, never on the command line
    cmd += [image, "solve", "--task-dir", "/input", "--out", "/app/output"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120)
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        rc, err = -9, "TIMEOUT"
    agent_s = time.time() - t0
    (dbg / "container_stderr.txt").write_text(err[-200000:], encoding="utf-8")
    res = {"unit": uid, "exit": rc, "agent_s": round(agent_s, 1)}
    try:
        summ = json.loads((dbg / "summary.json").read_text(encoding="utf-8"))
        res["requests"] = summ.get("house_requests")
        res["attempts"] = len(summ.get("attempts", []))
        res["best"] = summ.get("best")
    except Exception:  # noqa: BLE001
        pass
    res["out_files"] = ";".join(sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()))
    res.update(grade(unit, out, base))
    res["net_errors"] = net_errors(dbg)
    (base / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return res


NET_PAT = ("URLError", "Connection", "empty response", "RemoteDisconnected", "timed out", "network not reachable")


def net_errors(dbg: pathlib.Path) -> int:
    """House calls in this unit that failed for network reasons (not agent reasons)."""
    try:
        text = (dbg / "agent.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines()
               if ("house call" in line and "'ok': False" in line or "network not reachable" in line)
               and any(k in line for k in NET_PAT))


def grade(unit: pathlib.Path, out: pathlib.Path, base: pathlib.Path) -> dict:
    """Run the unit's real checker against a COPY of the outputs (the checker writes reward files)."""
    gout = base / "graded"
    if gout.exists():
        shutil.rmtree(gout)
    shutil.copytree(out, gout)
    os.chmod(gout, 0o777)
    vlog = base / "verifier_log"
    vlog.mkdir(exist_ok=True)
    mounts = ["-v", f"{dpath(unit)}:/input:ro", "-v", f"{dpath(unit / 'checks')}:/tests:ro",
              "-v", f"{dpath(gout)}:/app/output", "-v", f"{dpath(gout)}:/output",
              "-v", f"{dpath(vlog)}:/logs/verifier"]  # keep the verifier's diagnostic.json
    seen = {"/input", "/tests", "/app/output", "/output"}
    pmap = build_path_map(unit)
    for img, real in sorted(pmap.items(), key=lambda kv: len(kv[0])):
        if img in seen or img in ("/app", "/tests") or img.startswith(("/app/output", "/output", "/input")):
            continue  # never mount over /app itself: /app/output and /app/<file> mounts live inside it
        if any(img.startswith(s.rstrip("/") + "/") for s in seen if s not in ("/app/output", "/output")):
            continue  # already covered by a parent directory mount
        mounts += ["-v", f"{dpath(pathlib.Path(real))}:{img}:ro"]
        seen.add(img)
    cmd = ["docker", "run", "--rm", "--network=none", "-e", "OUTPUT_DIR=/app/output", *mounts,
           "finance-bench-sandbox:latest", "bash", "/tests/test.sh"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        log = p.stdout[-20000:] + p.stderr[-5000:]
    except subprocess.TimeoutExpired:
        log = "GRADER TIMEOUT"
    (base / "grader_log.txt").write_text(log, encoding="utf-8")
    res = {"reward": 0.0, "passed": 0, "failed": 0, "failing": ""}
    try:
        res["reward"] = float(json.loads((gout / "reward.json").read_text())["reward"])
    except Exception:  # noqa: BLE001
        pass
    try:
        rep = json.loads((gout / "pytest_report.json").read_text())
        tests = rep.get("tests", [])
        res["passed"] = sum(t["outcome"] == "passed" for t in tests)
        bad = [t["nodeid"].split("::")[-1] for t in tests if t["outcome"] in ("failed", "error")]
        res["failed"] = len(bad)
        res["failing"] = ";".join(bad[:12])
    except Exception:  # noqa: BLE001
        pass
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="t1-agent:dev")
    ap.add_argument("--units", default="panel")
    ap.add_argument("--show-checks", action="store_true", help="let the agent see checks/ (NOT platform-like)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the agent container")
    ap.add_argument("--mock", action="store_true", help="use dev/mock_house.py on host port 8765 instead of a real model")
    ap.add_argument("--resume", action="store_true", help="keep units already graded in this run tag")
    ap.add_argument("--net-reruns", type=int, default=1, help="rerun units that failed with network errors")
    a = ap.parse_args()
    if a.units == "panel":
        units = PANEL
    elif a.units == "all":
        units = sorted(p.name for p in (KIT / "units").iterdir())
    else:
        units = a.units.split(",")
    extra = dict(e.split("=", 1) for e in a.env)
    if a.mock:
        extra.update({"MODEL_ENDPOINT": "http://host.docker.internal:8765", "MODEL_NAME": "mock", "MODEL_TOKEN": "mock"})
    tag = a.tag or time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={'configured' if model_env() else 'NONE (offline)'} units={len(units)} run={run_dir}")
    done: dict[str, dict] = {}
    if a.resume:
        for u in units:
            rj = run_dir / u / "result.json"
            if rj.exists():
                try:
                    done[u] = json.loads(rj.read_text(encoding="utf-8"))
                except ValueError:
                    pass
        print(f"resume: {len(done)} unit(s) already graded")

    def run_many(todo: list[str]) -> None:
        with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(run_unit, a.image, KIT / "units" / u, run_dir, a.show_checks, extra): u for u in todo}
            for f in cf.as_completed(futs):
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001
                    r = {"unit": futs[f], "reward": 0.0, "error": f"{type(e).__name__}: {e}"}
                done[r["unit"]] = r
                print(f"{r.get('reward', 0):.0f} {r['unit'][:44]:44s} t={r.get('agent_s', '?')}s req={r.get('requests')} "
                      f"net={r.get('net_errors', 0)} pass={r.get('passed')} fail={r.get('failed')} "
                      f"{str(r.get('failing', ''))[:80]}", flush=True)

    run_many([u for u in units if u not in done])
    for _ in range(a.net_reruns):
        again = [u for u in units if float(done.get(u, {}).get("reward") or 0) < 1 and int(done.get(u, {}).get("net_errors") or 0) > 0]
        if not again:
            break
        print(f"network reruns: {again}")
        run_many(again)
    rows = [done[u] for u in units if u in done]
    keys = sorted({k for r in rows for k in r})
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["unit"]))
    n = len(rows)
    s = sum(r.get("reward", 0) for r in rows)
    t = sum(r.get("agent_s", 0) or 0 for r in rows)
    print(f"PASS@1 = {s:.0f}/{n} = {s / max(n, 1):.3f}   mean agent time {t / max(n, 1):.0f}s")


if __name__ == "__main__":
    main()
