"""Download the public benchmarks into data/raw and verify them against data/raw_manifest.json.

Usage: python data/download.py            (skips files that are already present and verified)

The manifest records the source URL and the SHA-256 of every file as used in the paper.
A checksum mismatch means that the source has changed since; the file is kept, and the
mismatch is reported, because the processed splits may then differ from the paper's.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads((HERE / "raw_manifest.json").read_text(encoding="utf-8"))
    changed = []
    for rel, entry in manifest.items():
        path = ROOT / rel
        if path.exists() and sha256(path) == entry["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel}")
        req = urllib.request.Request(entry["url"], headers={"User-Agent": "LLMJudgementFusion"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
        if sha256(path) != entry["sha256"]:
            changed.append(rel)
    for rel in changed:
        print(f"checksum differs from the paper's copy: {rel}")
    print(f"{len(manifest)} files in data/raw, {len(changed)} differing")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
