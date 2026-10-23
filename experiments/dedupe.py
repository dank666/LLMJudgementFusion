"""Find duplicate items across the splits and write data/processed/excluded.json.

Two items are duplicates when passage, stem and options agree after removing case, spaces and punctuation.
The first one is kept, development splits before test splits, so no test item repeats a development item.
Groups whose answer keys disagree are dropped entirely.
"""

import json
import re

from common import EXCLUDED_PATH, load_split

SPLITS = ["intl_dev", "multi_dev", "intl_test", "gaokao", "multi_test"]


def norm(s):
    return re.sub(r"\W+", "", (s or "").lower())


def key(q):
    return norm(q.passage) + "#" + norm(q.stem) + "|" + "|".join(norm(q.options[o]) for o in sorted(q.options))


def main():
    groups = {}
    for sp in SPLITS:
        for q in load_split(sp, exclude=False):
            groups.setdefault(key(q), []).append(q)
    excluded = {}
    for qs in groups.values():
        if len(qs) == 1:
            continue
        if len({tuple(sorted(q.answer)) for q in qs}) > 1:
            excluded.update({q.uid: "duplicates with conflicting answer keys" for q in qs})
        else:
            excluded.update({q.uid: f"duplicate of {qs[0].uid}" for q in qs[1:]})
    EXCLUDED_PATH.write_text(json.dumps(dict(sorted(excluded.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(excluded), "items excluded")


if __name__ == "__main__":
    main()
