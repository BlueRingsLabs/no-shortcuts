"""Freeze the lesson text that the course's own models and indexes are built from.

    python tools/snapshot_course.py

Several labs train on this course (the tiny GPT, the tokenizer, the n-gram and RNN labs, the retrieval encoder) or
search it (Module 41 onward). They read Parts 0 to V from `part-5-language-models/_shared/course_snapshot.json.gz`,
not from the live lesson files, for the same reason any training set should be versioned (48.2): so that fixing a
typo in a lesson doesn't retrain the models and move every number quoted in Parts V and VI.

Rerunning this script is a deliberate data release. It changes the shared models' cache keys, and every lab that
depends on them will produce slightly different numbers, which the lessons quote. Rerun those labs and update the
lessons in the same change.
"""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part-5-language-models" / "_shared"))
from corpus import SNAPSHOT_PATH, SNAPSHOT_PARTS, strip_asides  # noqa: E402


def main():
    files = sorted(p for p in ROOT.glob("part-*/*/[0-9][0-9].[0-9]*.md") if int(p.parts[-3].split("-")[1]) in SNAPSHOT_PARTS)
    snap = {str(p.relative_to(ROOT)): strip_asides(p.read_text(encoding="utf-8")) for p in files}
    snap["course.yaml"] = (ROOT / "course.yaml").read_text(encoding="utf-8")   # the instruction data of 38.x comes from it
    data = json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=0).encode("utf-8")
    with open(SNAPSHOT_PATH, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        f.write(data)                                                           # mtime=0: same text, same bytes
    print(f"{len(snap)} lessons, {len(data) / 1e6:.2f} MB of text -> {SNAPSHOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
