"""Profile the public Development units (the development distribution).

Usage: python dev/survey_units.py <units_dir> [--csv out.csv]
"""
from __future__ import annotations

import collections
import csv
import os
import pathlib
import re
import sys
import tomllib


def main() -> None:
    units = pathlib.Path(sys.argv[1])
    out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    rows = []
    ext = collections.Counter()
    cats = collections.Counter()
    diff = collections.Counter()
    for u in sorted(units.iterdir()):
        card = tomllib.loads((u / "card.toml").read_text(encoding="utf8"))
        ins = (u / "instruction.md").read_text(encoding="utf8")
        outs = sorted(set(re.findall(r"/(?:app/)?output/([\w\-./]+\.\w+)", ins)))
        outs = [o for o in outs if o not in ("reward.json", "pytest_report.json")]
        files, total = [], 0
        for p in u.rglob("*"):
            if not p.is_file() or "checks" in p.relative_to(u).parts:
                continue
            if p.name in ("card.toml", "instruction.md", "manifest.json", "Dockerfile"):
                continue
            files.append(p)
            total += p.stat().st_size
            ext[p.suffix] += 1
        m = card["metadata"]
        cats[m["category"]] += 1
        diff[m["difficulty"]] += 1
        rows.append(
            dict(
                unit=u.name,
                category=m["category"],
                difficulty=m["difficulty"],
                timeout=int(card["agent"]["timeout_sec"]),
                cutoff=card["provenance"].get("data_cutoff", ""),
                instr_chars=len(ins),
                n_files=len(files),
                kb=total // 1024,
                outputs=";".join(outs),
                n_checks=len(re.findall(r"^def test_", (u / "checks/test_outputs.py").read_text(encoding="utf8"), re.M))
                if (u / "checks/test_outputs.py").exists()
                else -1,
            )
        )
    for r in rows:
        print(
            f"{r['unit'][:44]:44s} {r['category'][:16]:16s} {r['difficulty'][:4]:4s} "
            f"{r['timeout']:5d} {r['instr_chars']:6d} {r['n_files']:3d} {r['kb']:7d}K "
            f"t{r['n_checks']:3d} {r['outputs'][:70]}"
        )
    print(cats)
    print(diff)
    print(ext.most_common())
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
