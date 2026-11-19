"""Build assets/casebook from benchmark runs (rule 8: our own public-unit solutions, used only
as examples for OTHER units).

Only solutions that our agent produced AND that passed the unit's real checker are admitted.
Each case stores: our generated code (prelude removed, every GUID-like string removed), the
unit's category/title, and an instruction token vector used for similarity-based exclusion.
No unit data is copied.

    python dev/build_casebook.py benchmarks/runs/<tag> [benchmarks/runs/<tag2> ...]
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "track1-coding-public" / "units"
sys.path.insert(0, str(ROOT))
from agent.casebook import CASEBOOK_DIR, _tokens  # noqa: E402
from agent.codeio import strip_prelude  # noqa: E402
from agent.task import GUID_RE, scrub  # noqa: E402

MAX_CODE = 9000


def main() -> None:
    cases: dict[str, dict] = {}
    idx = CASEBOOK_DIR / "index.json"
    if idx.exists():
        for c in json.loads(idx.read_text(encoding="utf-8")):
            cases[c["unit_id"]] = c
    for run in sys.argv[1:]:
        run = pathlib.Path(run)
        rows = list(csv.DictReader(open(run / "results.csv", encoding="utf-8")))
        for r in rows:
            if float(r.get("reward") or 0) < 1 or not r.get("best"):
                continue
            uid = r["unit"]
            src = run / uid / "debug" / f"solution_{int(float(r['best']))}.py"
            if not src.exists():
                continue
            code = strip_prelude(src.read_text(encoding="utf-8"))
            code = GUID_RE.sub("", code)
            code = re.sub(r"(?im)^.*canary.*$\n?", "", code)
            if len(code) > MAX_CODE:
                continue  # very long programs make poor examples and cost prompt budget
            card = tomllib.loads((KIT / uid / "card.toml").read_text(encoding="utf-8"))
            instr = scrub((KIT / uid / "instruction.md").read_text(encoding="utf-8"))
            cat = card.get("metadata", {}).get("category", "")
            rel = f"{re.sub(r'[^a-z0-9_-]', '_', cat.lower()) or 'misc'}/{uid}.py"
            dst = CASEBOOK_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(code, encoding="utf-8")
            toks = _tokens(instr)
            cases[uid] = {
                "unit_id": uid,
                "title": card.get("task", {}).get("title", uid),
                "category": cat,
                "file": rel,
                "notes": "",
                "tokens": dict(toks.most_common(300)),
            }
    CASEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(sorted(cases.values(), key=lambda c: c["unit_id"]), indent=0), encoding="utf-8")
    # final contamination sweep over the whole casebook
    for p in CASEBOOK_DIR.rglob("*"):
        if p.is_file() and GUID_RE.search(p.read_text(encoding="utf-8", errors="replace")):
            if p.name != "index.json":
                raise SystemExit(f"GUID found in {p}; refusing")
    print(f"casebook: {len(cases)} cases")


if __name__ == "__main__":
    main()
