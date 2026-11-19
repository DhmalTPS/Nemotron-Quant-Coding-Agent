# SPDX-License-Identifier: Apache-2.0
"""Entrypoint: `solve --task-dir /input --out /app/output`.

Contract: consume the verb, never crash, always leave a well-formed output tree, exit 0.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback


class _HardStop(BaseException):
    pass


def _alarm(_signum, _frame):  # noqa: ANN001
    raise _HardStop()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="solve", description="Agenthon 2026 Track 1 agent")
    parser.add_argument("verb", nargs="?", default="solve")
    parser.add_argument("--task-dir", default="/input")
    parser.add_argument("--out", default="/app/output")
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        args = argparse.Namespace(verb="solve", task_dir="/input", out="/app/output")
    if args.verb != "solve":
        print(f"unsupported verb {args.verb!r}; treating as solve", file=sys.stderr)

    from .config import Config
    from .log import log, set_debug_dir

    cfg = Config()
    set_debug_dir(cfg.debug_dir)
    controller = None
    try:
        from .controller import Controller

        controller = Controller(args.task_dir, args.out, cfg)
        # safety net: stop well before the card ceiling even if something hangs
        try:
            import tomllib

            with open(os.path.join(args.task_dir, "card.toml"), "rb") as fh:
                card_to = float(tomllib.load(fh).get("agent", {}).get("timeout_sec", 1800))
        except Exception:  # noqa: BLE001
            card_to = 1800.0
        hard = int(max(60, min(card_to - 25, cfg.max_unit_seconds + 120)))
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(hard)
        controller.run()
    except _HardStop:
        log("hard stop: wall-clock safety net fired")
    except BaseException as e:  # noqa: BLE001
        log(f"agent error: {type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}")
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        try:
            if controller is not None:
                controller.finish()
        except BaseException as e:  # noqa: BLE001
            log(f"finish error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
