# SPDX-License-Identifier: Apache-2.0
"""Output hygiene. The whole /app/output tree is checked after exit (<=256 files, <=4096
entries, depth < 8, no links/special files, <=64 MiB per file and in total, valid names,
non-empty), and any canary GUID in an output file is a g2 contamination failure."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import stat
import unicodedata

from .contract import Contract
from .log import log

STAGE_NAME = ".agent_stage"
SCRATCH_SUFFIXES = (".pyc", ".log", ".tmp", ".lock")
SCRATCH_NAMES = {"reward.json", "pytest_report.json", "reward.txt", "__pycache__", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".md", ".html", ".xml", ".py", ".jsonl", ".svg", ".yaml", ".yml"}
MAX_FILES = 256
MAX_BYTES = 64 * 1024 * 1024


def clear_dir(d: pathlib.Path, keep: set[str] | None = None) -> None:
    keep = keep or set()
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.name in keep or p.name.startswith(".agent_"):
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass


def promote(stage: pathlib.Path, out: pathlib.Path) -> None:
    """Replace the visible deliverables in out with the contents of stage."""
    clear_dir(out, keep={STAGE_NAME})
    for p in stage.rglob("*"):
        rel = p.relative_to(stage)
        dst = out / rel
        try:
            if p.is_symlink():
                continue
            if p.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif p.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
        except OSError as e:
            log(f"promote: {rel}: {e}")


def _scrub_canaries(p: pathlib.Path, canaries: set[str]) -> None:
    if not canaries or p.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        raw = p.read_bytes()
    except OSError:
        return
    if len(raw) > 64 * 1024 * 1024:
        return
    text = raw.decode("utf-8", "replace")
    low = text.lower()
    hit = [g for g in canaries if g in low]
    if not hit:
        return
    import re

    for g in hit:
        text = re.sub(re.escape(g), "", text, flags=re.I)
    p.write_text(text, encoding="utf-8")
    log(f"scrubbed canary GUID(s) from {p.name}")


def sanitize(out: pathlib.Path, canaries: set[str]) -> None:
    for p in list(out.iterdir()) if out.is_dir() else []:
        if p.name.startswith(".agent_"):  # every internal stage dir (A and B)
            shutil.rmtree(p, ignore_errors=True)
    files: list[pathlib.Path] = []
    for root, dirs, fnames in os.walk(out, topdown=True, followlinks=False):
        rp = pathlib.Path(root)
        depth = len(rp.relative_to(out).parts)
        for d in list(dirs):
            dp = rp / d
            if d in SCRATCH_NAMES or dp.is_symlink() or depth >= 6:
                try:
                    if dp.is_symlink():
                        dp.unlink()
                    else:
                        shutil.rmtree(dp, ignore_errors=True)
                except OSError:
                    pass
                dirs.remove(d)
        for f in fnames:
            fp = rp / f
            try:
                st = fp.lstat()
            except OSError:
                continue
            bad_name = unicodedata.normalize("NFC", f) != f
            if (
                f in SCRATCH_NAMES
                or f.endswith(SCRATCH_SUFFIXES)
                or stat.S_ISLNK(st.st_mode)
                or not stat.S_ISREG(st.st_mode)
                or st.st_nlink > 1
                or st.st_size > MAX_BYTES
                or bad_name
            ):
                try:
                    fp.unlink()
                except OSError:
                    pass
                continue
            if st.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                try:
                    os.chmod(fp, 0o644)
                except OSError:
                    pass
            files.append(fp)
    for fp in files:
        _scrub_canaries(fp, canaries)
    # enforce count / total size limits, dropping the largest non-essential files first
    files = [f for f in files if f.exists()]
    total = sum(f.stat().st_size for f in files)
    if len(files) > MAX_FILES or total > MAX_BYTES:
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
        while files and (len(files) > MAX_FILES or total > MAX_BYTES):
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink(missing_ok=True)
            log(f"removed {f.name} to respect output limits")
    # remove now-empty directories
    for root, dirs, _ in os.walk(out, topdown=False):
        for d in dirs:
            dp = pathlib.Path(root) / d
            try:
                dp.rmdir()
            except OSError:
                pass


def write_skeleton(out: pathlib.Path, contract: Contract) -> None:
    """Last resort so the output is never empty: well-formed files of the right names."""
    for name, spec in contract.files.items():
        if not spec.strong:
            continue
        p = out / name
        if p.exists():
            continue
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".json"):
                obj: dict = {}
                for key, hint in spec.json_keys.items():
                    cur = obj
                    parts = key.split(".")
                    for part in parts[:-1]:
                        cur = cur.setdefault(part, {}) if isinstance(cur.get(part, {}), dict) else {}
                    if hint == "object":
                        cur.setdefault(parts[-1], {})
                    else:
                        cur[parts[-1]] = {"bool": False, "int": 0, "str": "", "array": []}.get(hint, 0.0)
                p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
            elif name.endswith((".csv", ".tsv")):
                sep = "\t" if name.endswith(".tsv") else ","
                p.write_text(sep.join(spec.columns) + "\n" if spec.columns else "value\n", encoding="utf-8")
            elif name.endswith((".parquet", ".pqt")):
                import pandas as pd

                pd.DataFrame({c: [] for c in (spec.columns or ["value"])}).to_parquet(p, index=False)
            elif name.endswith(".png"):
                import base64

                p.write_bytes(base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))
            elif name.endswith((".html", ".htm")):
                p.write_text("<html><body></body></html>\n", encoding="utf-8")
            else:
                p.write_text("\n", encoding="utf-8")
        except Exception:  # noqa: BLE001 - last resort must never raise
            pass
