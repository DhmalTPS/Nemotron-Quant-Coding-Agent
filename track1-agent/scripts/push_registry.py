"""Resumable, chunked image push to Docker Hub from the HOST (bypasses the Docker Desktop VM network).

    python scripts/push_registry.py t1-agent:v5 <tag>

* `docker save` -> OCI archive on F: (once), pick the linux/amd64 image manifest
* every blob: skip if already present (HEAD), else POST + PATCH 16 MiB chunks with retries; an
  interrupted upload resumes from the server-reported Range
* PUT the manifest bytes unchanged, so the pushed digest equals the local manifest digest
Credentials: extra_info/keys/registry_user.txt + registry_token.txt (read here, never printed).
"""
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = ROOT.parent / "extra_info" / "keys"
WORK = ROOT / "benchmarks" / "push_work"
REG = "https://registry-1.docker.io"
CHUNK = 16 * 1024 * 1024


class Hub:
    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.user = (KEYS / "registry_user.txt").read_text(encoding="utf-8").strip()
        self.secret = (KEYS / "registry_token.txt").read_text(encoding="utf-8").strip()
        self.tok = ""
        self.tok_t = 0.0

    def token(self) -> str:
        if self.tok and time.time() - self.tok_t < 200:
            return self.tok
        url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{self.repo}:pull,push"
        basic = base64.b64encode(f"{self.user}:{self.secret}".encode()).decode()
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {basic}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.tok = json.loads(r.read())["token"]
        self.tok_t = time.time()
        return self.tok

    def call(self, method: str, url: str, data: bytes | None = None, headers: dict | None = None,
             ok=(200, 201, 202, 204, 307), tries: int = 8):
        if not url.startswith("http"):
            url = REG + url
        last = None
        for i in range(tries):
            h = {"Authorization": f"Bearer {self.token()}"}
            h.update(headers or {})
            req = urllib.request.Request(url, data=data, method=method, headers=h)
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.status, dict(r.headers), r.read()
            except urllib.error.HTTPError as e:
                if e.code in ok:
                    return e.code, dict(e.headers), e.read()
                if e.code == 401:
                    self.tok = ""
                if e.code == 404 and method == "HEAD":
                    return 404, {}, b""
                last = f"HTTP {e.code} {e.read()[:200]!r}"
                if e.code in (400, 403, 405) and i >= 1:
                    break
            except Exception as e:  # noqa: BLE001 - timeouts/resets
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 3 * 2 ** i))
        raise RuntimeError(f"{method} {url[:90]} failed: {last}")


def blob_exists(hub: Hub, digest: str) -> bool:
    code, _, _ = hub.call("HEAD", f"/v2/{hub.repo}/blobs/{digest}", ok=(200, 307), tries=4)
    return code in (200, 307)


def upload_blob(hub: Hub, digest: str, path: pathlib.Path) -> None:
    size = path.stat().st_size
    if blob_exists(hub, digest):
        print(f"  {digest[7:19]} exists ({size / 1e6:.1f} MB)")
        return
    _, h, _ = hub.call("POST", f"/v2/{hub.repo}/blobs/uploads/", data=b"", headers={"Content-Length": "0"})
    loc = h.get("Location") or h.get("location")
    offset = 0
    with open(path, "rb") as fh:
        while offset < size:
            fh.seek(offset)
            chunk = fh.read(CHUNK)
            end = offset + len(chunk) - 1
            try:
                _, h, _ = hub.call("PATCH", loc, data=chunk, tries=3, headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Range": f"{offset}-{end}", "Content-Length": str(len(chunk))})
                loc = h.get("Location") or h.get("location") or loc
                rng = h.get("Range") or h.get("range")
                offset = int(rng.split("-")[1]) + 1 if rng else end + 1
            except RuntimeError as e:
                # ask the server how much it has, then resume from there
                print(f"  chunk at {offset} failed ({str(e)[:80]}); querying upload status")
                code, h2, _ = hub.call("GET", loc, ok=(204,), tries=4)
                rng = h2.get("Range") or h2.get("range")
                offset = int(rng.split("-")[1]) + 1 if rng else 0
                loc = h2.get("Location") or h2.get("location") or loc
            print(f"  {digest[7:19]} {offset / 1e6:7.1f}/{size / 1e6:.1f} MB", flush=True)
    sep = "&" if "?" in loc else "?"
    hub.call("PUT", f"{loc}{sep}digest={urllib.parse.quote(digest)}", data=b"", headers={"Content-Length": "0"})
    print(f"  {digest[7:19]} uploaded")


def main() -> None:
    image, tag = sys.argv[1], sys.argv[2]
    user = (KEYS / "registry_user.txt").read_text(encoding="utf-8").strip()
    repo = f"{user}/agenthon-t1"
    WORK.mkdir(parents=True, exist_ok=True)
    tar = WORK / "image.tar"
    print("docker save ...", flush=True)
    subprocess.run(["docker", "save", image, "-o", str(tar)], check=True)
    out = WORK / "oci"
    if out.exists():
        import shutil

        shutil.rmtree(out)
    with tarfile.open(tar) as tf:
        tf.extractall(out, filter="data")
    blobs = out / "blobs" / "sha256"

    def blob(d: str) -> bytes:
        return (blobs / d.split(":")[1]).read_bytes()

    idx = json.loads((out / "index.json").read_text())
    stack = list(idx["manifests"])
    manifest_desc = None
    while stack:
        d = stack.pop(0)
        mt = d.get("mediaType", "")
        if "index" in mt or "manifest.list" in mt:
            stack.extend(json.loads(blob(d["digest"]))["manifests"])
            continue
        plat = d.get("platform") or {}
        if plat.get("architecture", "amd64") == "amd64" and plat.get("os", "linux") == "linux":
            m = json.loads(blob(d["digest"]))
            if m.get("config", {}).get("mediaType", "").endswith(("image.config.v1+json", "container.image.v1+json")):
                manifest_desc = d
                break
    if manifest_desc is None:
        raise SystemExit("no linux/amd64 image manifest in the archive")
    mbytes = blob(manifest_desc["digest"])
    assert "sha256:" + hashlib.sha256(mbytes).hexdigest() == manifest_desc["digest"]
    manifest = json.loads(mbytes)
    hub = Hub(repo)
    items = [manifest["config"]] + manifest["layers"]
    print(f"pushing {len(items)} blobs, {sum(i['size'] for i in items) / 1e9:.2f} GB to docker.io/{'<user>'}/agenthon-t1")
    for it in items:
        upload_blob(hub, it["digest"], blobs / it["digest"].split(":")[1])
    mt = manifest_desc.get("mediaType") or manifest.get("mediaType")
    hub.call("PUT", f"/v2/{repo}/manifests/{tag}", data=mbytes, headers={"Content-Type": mt})
    print(f"PUSHED digest={manifest_desc['digest']} tag={tag}")
    (WORK / "last_digest.txt").write_text(manifest_desc["digest"], encoding="utf-8")


if __name__ == "__main__":
    main()
