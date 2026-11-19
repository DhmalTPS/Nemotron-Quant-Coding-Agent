"""Verify that an image is pullable ANONYMOUSLY by digest (no saved credentials involved).

    python scripts/check_public.py docker.io USER/agenthon-t1 sha256:<digest>

Docker Hub: anonymous token from auth.docker.io, then GET the manifest (index or single) by digest
and the config blob of the linux/amd64 manifest. Exit 0 only if everything answers 200.
GHCR: same flow via ghcr.io/token.
"""
from __future__ import annotations

import json
import sys
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def _get(url: str, token: str, accept: str = ACCEPT, head: bool = False) -> tuple[int, bytes, dict]:
    req = urllib.request.Request(url, method="HEAD" if head else "GET",
                                 headers={"Authorization": f"Bearer {token}", "Accept": accept})
    # no cookies / netrc / docker config are consulted by urllib: this is a credential-free client
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, b"" if head else r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, b"", {}


def main() -> None:
    registry, repo, digest = sys.argv[1], sys.argv[2], sys.argv[3]
    if registry in ("docker.io", "registry-1.docker.io", "index.docker.io"):
        tok_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        base = f"https://registry-1.docker.io/v2/{repo}"
    elif registry == "ghcr.io":
        tok_url = f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io"
        base = f"https://ghcr.io/v2/{repo}"
    else:
        raise SystemExit("unsupported registry")
    with urllib.request.urlopen(tok_url, timeout=60) as r:
        token = json.loads(r.read())["token"]
    code, body, hdr = _get(f"{base}/manifests/{digest}", token)
    print(f"manifest {digest[:19]}... -> HTTP {code}")
    if code != 200:
        raise SystemExit("NOT anonymously pullable (repository private, wrong digest, or not pushed)")
    doc = json.loads(body)
    manifest = doc
    if "manifests" in doc:  # index: pick linux/amd64
        amd = [m for m in doc["manifests"] if (m.get("platform") or {}).get("architecture") == "amd64"
               and (m.get("platform") or {}).get("os") == "linux"]
        if not amd:
            raise SystemExit("index has no linux/amd64 manifest")
        code, body, _ = _get(f"{base}/manifests/{amd[0]['digest']}", token)
        print(f"linux/amd64 manifest -> HTTP {code}")
        if code != 200:
            raise SystemExit("amd64 manifest not anonymously readable")
        manifest = json.loads(body)
    cfg = manifest["config"]["digest"]
    code, _, _ = _get(f"{base}/blobs/{cfg}", token, accept="*/*", head=True)
    print(f"config blob -> HTTP {code}")
    layers = manifest.get("layers", [])
    bad = 0
    for l in layers:
        c, _, _ = _get(f"{base}/blobs/{l['digest']}", token, accept="*/*", head=True)
        bad += c != 200
    print(f"layers: {len(layers)} checked, {bad} not readable")
    if code != 200 or bad:
        raise SystemExit("image blobs not anonymously readable")
    print("OK: anonymously pullable by digest")


if __name__ == "__main__":
    main()
