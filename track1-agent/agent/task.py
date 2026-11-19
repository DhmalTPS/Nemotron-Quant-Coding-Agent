# SPDX-License-Identifier: Apache-2.0
"""Deterministic task ingestion: instruction, card, file inventory, path map, canaries.

Facts this module encodes (verified against the public kit):
* The whole unit directory is mounted read-only at --task-dir (normally /input).
* 40/86 instructions refer to inputs at their *unit-image* location (/app/data/..., /app/x.csv),
  which comes from the unit's environment/Dockerfile COPY lines. At evaluation time the files
  live under <task-dir>/environment/. We rebuild that mapping.
* checks/ may or may not be mounted (it is sealed on the platform); never depend on it.
* Canary GUIDs sit in instruction.md and checks; a canary in any output file is a g2 failure.
"""

from __future__ import annotations

import os
import pathlib
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any

GUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CANARY_LINE_RE = re.compile(r"^.*(canary|BENCHMARK DATA SHOULD NEVER).*$\n?", re.I | re.M)
META_FILES = {"instruction.md", "card.toml", "manifest.json"}


@dataclass
class InputFile:
    path: pathlib.Path  # real path inside the container
    rel: str  # relative to task dir
    size: int
    image_paths: list[str] = field(default_factory=list)  # where the unit image would put it


@dataclass
class Task:
    task_dir: pathlib.Path
    out_dir: pathlib.Path
    unit_id: str
    instruction: str  # canary-scrubbed
    card: dict[str, Any]
    files: list[InputFile]
    path_map: dict[str, str]  # image path (file or dir) -> real path
    checks_text: str  # scrubbed checker source if mounted, else ""
    canaries: set[str]
    category: str = ""
    difficulty: str = ""
    timeout_sec: float = 1800.0
    cpus: int = 0
    gpu: bool = False

    @property
    def data_dir(self) -> pathlib.Path:
        d = self.task_dir / "environment" / "data"
        return d if d.is_dir() else self.task_dir / "environment"


def scrub(text: str) -> str:
    """Remove canary banner lines and any GUID adjacent to the word canary."""
    text = CANARY_LINE_RE.sub("", text)
    return text


def _read(p: pathlib.Path, limit: int = 400_000) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _parse_copies(dockerfile: pathlib.Path) -> list[tuple[str, str]]:
    """Return (src relative to environment/, absolute dst) pairs from COPY/ADD lines."""
    out: list[tuple[str, str]] = []
    text = _read(dockerfile)
    workdir = "/"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"WORKDIR\s+(\S+)", line, re.I)
        if m:
            workdir = m.group(1)
            continue
        m = re.match(r"(COPY|ADD)\s+(.*)", line, re.I)
        if not m:
            continue
        parts = [p for p in m.group(2).split() if not p.startswith("--")]
        if parts and parts[0].startswith("["):
            try:
                import json

                parts = json.loads(m.group(2))
            except Exception:  # noqa: BLE001
                continue
        if len(parts) < 2:
            continue
        *srcs, dst = parts
        if not dst.startswith("/"):
            dst = workdir.rstrip("/") + "/" + dst
        for s in srcs:
            out.append((s.strip("./") if s not in (".", "./") else "", dst))
    return out


def build_path_map(task_dir: pathlib.Path) -> dict[str, str]:
    env = task_dir / "environment"
    pmap: dict[str, str] = {}
    for src, dst in _parse_copies(env / "Dockerfile"):
        real = env / src if src else env
        if not real.exists():
            continue
        if real.is_dir():
            # "COPY data/ /app/data/" -> dir mapping; "COPY data/ /app/" -> files land in /app
            pmap[dst.rstrip("/")] = str(real)
        else:
            if dst.endswith("/"):
                pmap[dst + real.name] = str(real)
            else:
                pmap[dst] = str(real)
    data = env / "data"
    if data.is_dir():
        pmap.setdefault("/app/data", str(data))
        for p in data.iterdir():
            if p.is_file():
                pmap.setdefault(f"/app/{p.name}", str(p))
                pmap.setdefault(f"/app/data/{p.name}", str(p))
    pmap.setdefault("/input", str(task_dir))
    return pmap


def _image_paths_for(real: pathlib.Path, pmap: dict[str, str]) -> list[str]:
    res = []
    rs = str(real)
    for img, tgt in pmap.items():
        if img == "/input":
            continue
        if rs == tgt:
            res.append(img)
        elif rs.startswith(tgt.rstrip("/") + "/"):
            res.append(img.rstrip("/") + "/" + rs[len(tgt.rstrip("/")) + 1 :])
    return sorted(set(res))


def load_task(task_dir: str, out_dir: str, use_checks: bool = True) -> Task:
    td = pathlib.Path(task_dir).absolute()
    od = pathlib.Path(out_dir).absolute()
    raw_instr = _read(td / "instruction.md")
    card: dict[str, Any] = {}
    try:
        card = tomllib.loads(_read(td / "card.toml"))
    except Exception:  # noqa: BLE001
        card = {}
    canaries = set()
    for m in GUID_RE.finditer(raw_instr):
        canaries.add(m.group(0).lower())
    cg = (card.get("contamination") or {}).get("canary_guid")
    if cg:
        canaries.add(str(cg).lower())

    pmap = build_path_map(td)
    files: list[InputFile] = []
    for p in sorted(td.rglob("*")):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        rel = p.relative_to(td).as_posix()
        if rel in META_FILES or rel.startswith("checks/") or rel.endswith("Dockerfile"):
            continue
        if any(part.startswith(".") for part in p.relative_to(td).parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        files.append(InputFile(path=p, rel=rel, size=size, image_paths=_image_paths_for(p, pmap)))

    # canaries can also sit inside data files (e.g. a "_canary" field in params.json)
    for f in files:
        if f.size <= 2_000_000 and f.path.suffix.lower() in (".json", ".csv", ".txt", ".md", ".tsv", ".py", ".xml", ".html", ".jsonl"):
            for line in _read(f.path, 2_000_000).splitlines():
                if "canary" in line.lower():
                    for m in GUID_RE.finditer(line):
                        canaries.add(m.group(0).lower())

    checks_text = ""
    cdir = td / "checks"
    if cdir.is_dir():
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".sh", ".csv") and p.stat().st_size < 400_000:
                t = _read(p)
                for m in GUID_RE.finditer(t):
                    canaries.add(m.group(0).lower())
                if use_checks and p.suffix == ".py":
                    checks_text += f"\n# ===== {p.relative_to(td).as_posix()} =====\n" + scrub(t)

    task = card.get("task") or {}
    meta = card.get("metadata") or {}
    agent = card.get("agent") or {}
    envc = card.get("environment") or {}
    instr = scrub(raw_instr)
    for g in canaries:
        instr = re.sub(re.escape(g), "", instr, flags=re.I)
    try:
        timeout = float(agent.get("timeout_sec", 1800.0))
    except (TypeError, ValueError):
        timeout = 1800.0
    return Task(
        task_dir=td,
        out_dir=od,
        unit_id=str(task.get("id") or td.name),
        instruction=instr,
        card=card,
        files=files,
        path_map=pmap,
        checks_text=checks_text,
        canaries=canaries,
        category=str(meta.get("category", "")),
        difficulty=str(meta.get("difficulty", "")),
        timeout_sec=timeout,
        cpus=int(envc.get("cpus", 0) or 0),
        gpu=bool(envc.get("gpu", False)),
    )


def card_summary(t: Task) -> str:
    prov = t.card.get("provenance") or {}
    meta = t.card.get("metadata") or {}
    return (
        f"unit_id={t.unit_id}; category={t.category}; difficulty={t.difficulty}; "
        f"tags={meta.get('tags', [])}; data_cutoff={prov.get('data_cutoff', '') or 'none'}; "
        f"agent_timeout_sec={t.timeout_sec:.0f}; cpus={t.cpus or os.cpu_count()}; gpu={t.gpu}"
    )
