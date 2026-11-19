# SPDX-License-Identifier: Apache-2.0
"""Execute a generated solution in a subprocess with a hard timeout and bounded logs.

Platform facts: root FS read-only, /tmp is a 64 MiB noexec tmpfs (python can still *read and
interpret* scripts there), 256 PIDs, no HOME. Caches are pointed at /tmp; thread pools are
capped so BLAS + joblib cannot exhaust the PID limit.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _tail(b: bytes, n: int) -> str:
    s = b.decode("utf-8", "replace")
    if len(s) <= n:
        return s
    return f"...[{len(s) - n} chars truncated]...\n" + s[-n:]


def run_solution(script: pathlib.Path, cwd: pathlib.Path, timeout: float, threads: int) -> RunResult:
    env = dict(os.environ)
    for k in ("MODEL_TOKEN", "MODEL_ENDPOINT", "MODEL_NAME"):
        env.pop(k, None)  # solutions never talk to the model
    tmp = "/tmp" if os.path.isdir("/tmp") else str(cwd)
    env.update(
        {
            "HOME": env.get("HOME") if env.get("HOME") not in (None, "", "/nonexistent") else tmp,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": os.path.join(tmp, "mpl"),
            "NUMBA_CACHE_DIR": os.path.join(tmp, "numba"),
            "XDG_CACHE_HOME": os.path.join(tmp, "cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_MAX_THREADS": str(threads),
            "LOKY_MAX_CPU_COUNT": str(threads),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    t0 = time.time()
    out_path = cwd / "_stdout.txt"
    err_path = cwd / "_stderr.txt"
    with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(cwd),
                env=env,
                stdout=fo,
                stderr=fe,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            return RunResult(-1, "", f"failed to start: {e}", 0.0, False)
        timed_out = False
        try:
            proc.wait(timeout=max(5.0, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
            proc.wait()
    so = out_path.read_bytes() if out_path.exists() else b""
    se = err_path.read_bytes() if err_path.exists() else b""
    for p in (out_path, err_path):
        try:
            p.unlink()
        except OSError:
            pass
    return RunResult(
        returncode=proc.returncode if not timed_out else -9,
        stdout=_tail(so, 5000),
        stderr=_tail(se, 6000),
        seconds=time.time() - t0,
        timed_out=timed_out,
    )
