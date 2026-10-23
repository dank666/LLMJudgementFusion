"""Check the prepared benchmark files against the checksums of the files used in the paper.

Usage: python data/check_processed.py   (after data/download.py and the experiments/prepare_*.py scripts)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    expected = json.loads((HERE / "processed_sha256.json").read_text(encoding="utf-8"))
    differing = []
    for rel, digest in expected.items():
        path = HERE / "processed" / rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            differing.append(rel)
    for rel in differing:
        print(f"missing or different from the paper's copy: {rel}")
    print(f"{len(expected)} prepared files, {len(differing)} differing")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
