"""Build, validate and pack the Track 1 submission.

    python scripts/package_submission.py --registry docker.io --repository USER/agenthon-t1 \
        --digest sha256:<64hex> [--phase dev|final]

Reads extra_info/keys/team_number.txt and passes extra_info/keys/team_key.txt to the toolkit
BY PATH (--team-key-file); the key is never printed, copied or put on a command line.
Writes submission/submission.json and submission/submission-<phase>-<digest12>.zip.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT.parent / "extra_info" / "keys"
OUT = ROOT / "submission"
HOUSE_MODEL = {
    "name": "nvidia/nemotron-3-super-120b-a12b",
    "version": "rl-030326-fp8",
    "revision": "rl-030326-fp8",
    "training_cutoff": "unpublished",
    "access": "api",
}


def find_qfbench2() -> str:
    import shutil

    exe = shutil.which("qfbench2")
    if exe:
        return exe
    import site

    for base in [site.getuserbase(), sys.prefix]:
        for sub in ("Scripts", "bin"):
            for name in ("qfbench2.exe", "qfbench2"):
                p = pathlib.Path(base) / sub / name
                if p.exists():
                    return str(p)
    for p in pathlib.Path(site.getuserbase()).glob("Python*/Scripts/qfbench2.exe"):
        return str(p)
    raise SystemExit("qfbench2 CLI not found; install qfbench2-common @ v2.5.1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--repository", required=True)
    ap.add_argument("--digest", required=True)
    ap.add_argument("--phase", default="dev", choices=["dev", "final"])
    a = ap.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", a.digest):
        raise SystemExit("digest must be sha256:<64 lowercase hex> (the pushed manifest digest)")
    team_number = (KEYS / "team_number.txt").read_text(encoding="utf-8").strip()
    key_file = KEYS / "team_key.txt"
    if not key_file.exists() or not team_number.isdigit():
        raise SystemExit("need extra_info/keys/team_key.txt and team_number.txt (digits)")

    from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest

    desc = {
        "schema_version": "1.1.0",
        "interface_version": "2.0",
        "competition_id": f"agenthon2026-coding-{a.phase}",
        "team_id": "team-placeholder",  # pack derives and rewrites the real id
        "track": "coding",
        "phase": a.phase,
        "category": "api",
        "image": {"registry": a.registry, "repository": a.repository, "digest": a.digest},
        "image_access": "public",
        "models": [HOUSE_MODEL],
        "license": "Apache-2.0",
        "descriptor_digest": "sha256:" + "0" * 64,
    }
    # The toolkit requires the key file to be mode 600, which NTFS cannot express. Run it inside a
    # throwaway Linux container (toolkit pinned to v2.5.1, network disabled): the key is copied,
    # CR/LF-stripped, into the container's private /tmp with mode 600 and never leaves it.
    def dp(p: pathlib.Path) -> str:
        return str(p.resolve()).replace("\\", "/")

    OUT.mkdir(exist_ok=True)
    keyprep = "tr -d '\\r\\n ' < /keys/team_key.txt > /tmp/k && chmod 600 /tmp/k && "
    alias = subprocess.run(
        ["docker", "run", "--rm", "--network=none", "-v", f"{dp(KEYS)}:/keys:ro", "--entrypoint", "sh",
         "qfb2-packer:2.5.1", "-c",
         keyprep + f"qfbench2 submission alias --team-number {team_number} --team-key-file /tmp/k; rm -f /tmp/k"],
        capture_output=True, text=True)
    m = re.search(r"team-[0-9a-f]{32}", alias.stdout + alias.stderr)
    if not m:
        raise SystemExit("could not derive team_id: " + (alias.stderr or alias.stdout)[-400:])
    desc["team_id"] = m.group(0)
    desc = seal_descriptor_digest(desc)
    SubmissionDescriptor.from_mapping(desc)  # raises on any schema/semantic problem
    dpath = OUT / "submission.json"
    dpath.write_text(json.dumps(desc, indent=2) + "\n", encoding="utf-8")
    zpath = OUT / f"submission-{a.phase}-{a.digest[7:19]}.zip"
    inner = (f"tr -d '\\r\\n ' < /keys/team_key.txt > /tmp/k && chmod 600 /tmp/k && "
             f"qfbench2 submission pack --team-number {team_number} --team-key-file /tmp/k "
             f"--descriptor /sub/submission.json --out /sub/{zpath.name} --force; rc=$?; rm -f /tmp/k; exit $rc")
    cmd = ["docker", "run", "--rm", "--network=none", "-v", f"{dp(KEYS)}:/keys:ro", "-v", f"{dp(OUT)}:/sub",
           "--entrypoint", "sh", "qfb2-packer:2.5.1", "-c", inner]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(r.returncode)
    final = json.loads(dpath.read_text(encoding="utf-8"))
    SubmissionDescriptor.from_mapping(final)
    print(f"OK: {zpath}  team_id={final['team_id']}  image={a.registry}/{a.repository}@{a.digest}")


if __name__ == "__main__":
    main()
