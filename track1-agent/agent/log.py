# SPDX-License-Identifier: Apache-2.0
"""Logging to stderr and (optionally) a debug directory. Never writes into the output dir."""

from __future__ import annotations

import os
import sys
import time

_T0 = time.time()
_DEBUG_DIR = ""


def set_debug_dir(path: str) -> None:
    global _DEBUG_DIR
    if path:
        try:
            os.makedirs(path, exist_ok=True)
            _DEBUG_DIR = path
        except OSError:
            _DEBUG_DIR = ""


def log(msg: str) -> None:
    line = f"[agent +{time.time() - _T0:7.1f}s] {msg}"
    try:
        print(line[:2000], file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass
    if _DEBUG_DIR:
        try:
            with open(os.path.join(_DEBUG_DIR, "agent.log"), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def save_debug(name: str, text: str) -> None:
    if not _DEBUG_DIR:
        return
    try:
        with open(os.path.join(_DEBUG_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError:
        pass
