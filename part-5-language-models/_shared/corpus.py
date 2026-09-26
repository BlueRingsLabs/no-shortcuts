"""Text for the Part V labs, from this course itself.

The labs run offline, so the corpus is the course's own lessons (Parts I to III: about a million characters of
English prose with math and code in it). That has a nice side effect: every model in Part V learns to sound a bit
like me, which you'll find is a good way to notice what it has and hasn't learned.

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
    from corpus import course_text, labeled_paragraphs, words, make_ner

course_text(parts=PARTS) -> str
    the lessons of the given parts, concatenated in a fixed order.
labeled_paragraphs(min_words=25) -> list of (paragraph, part_index)
    prose paragraphs (no code, tables or math blocks) with the Part they come from: 0 foundations, 1 data
    engineering, 2 classical ML. A small, real text-classification problem.
words(text) -> list of lowercase word tokens (letters, digits, apostrophes).
make_ner(n, seed=0, unseen_names=False, noise=0.15) -> list of (tokens, tags)
    synthetic sentences with PER / ORG / LOC entities in BIO tags, from templates and name lists; for sequence labeling.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = ("part-1-foundations", "part-2-data-engineering", "part-3-classical-ml")


ASIDE = re.compile(r"^> \*\*Scar tissue\.\*\*.*\n(?:>.*\n)*\n", re.M)


def strip_asides(text: str) -> str:
    """Remove the author's anecdote blocks ("> **Scar tissue.** ...") so every model and index in the course is built from
    the teaching text alone. Adding or editing an anecdote then changes no dataset, no cache key and no number."""
    return ASIDE.sub("", text)


SNAPSHOT_PATH = Path(__file__).resolve().parent / "course_snapshot.json.gz"
SNAPSHOT_PARTS = (0, 1, 2, 3, 4, 5)
_SNAPSHOT = None


def snapshot():
    """The frozen text of Parts 0 to V (tools/snapshot_course.py): what every lab that trains on or searches this course
    reads, so that editing a lesson never changes a dataset. Keys are paths relative to the repository root."""
    global _SNAPSHOT
    if _SNAPSHOT is None:
        import gzip
        import json
        with gzip.open(SNAPSHOT_PATH, "rt", encoding="utf-8") as f:
            _SNAPSHOT = json.load(f)
    return _SNAPSHOT


def read_lesson(path) -> str:
    """A lesson's text as the labs see it: from the snapshot when it's there, else the live file without anecdotes."""
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    snap = snapshot()
    return snap[rel] if rel in snap else strip_asides(Path(path).read_text(encoding="utf-8"))


def lesson_files(parts=PARTS):
    snap = snapshot()
    return [p for part in parts for p in sorted(ROOT / k for k in snap if k.startswith(part + "/"))]


def course_text(parts=PARTS) -> str:
    return "\n".join(read_lesson(p) for p in lesson_files(parts))


def _prose_paragraphs(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)                      # code blocks
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)                    # display math
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)          # exercise answers
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block[0] in "#|*-<>" or block[:2].isdigit() or block.startswith("1."):
            continue                                                        # headings, tables, lists, meta lines
        yield re.sub(r"\s+", " ", block)


def labeled_paragraphs(min_words=25):
    out = []
    for label, part in enumerate(PARTS):
        for f in lesson_files((part,)):
            for para in _prose_paragraphs(read_lesson(f)):
                if len(para.split()) >= min_words:
                    out.append((para, label))
    return out


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


_FIRST = ["Ana", "Luis", "Mei", "Omar", "Priya", "Jonas", "Chloe", "Kwame", "Sofia", "Dmitri", "Aiko", "Mateo", "Fatima",
          "Lars", "Ines", "Tomas", "Nadia", "Ravi", "Elena", "Hugo"]
_LAST = ["Garcia", "Chen", "Haddad", "Patel", "Novak", "Okafor", "Rossi", "Ivanova", "Tanaka", "Silva", "Kowalski",
         "Nguyen", "Fischer", "Mensah", "Moreau", "Larsen"]
_ORG = [["Northwind"], ["Acme", "Corp"], ["Globex"], ["Initech"], ["Umbrella", "Labs"], ["Stark", "Industries"],
        ["Blue", "Harbor", "Bank"], ["Vandelay", "Imports"], ["Hooli"], ["Cyberdyne", "Systems"]]
_LOC = [["Lisbon"], ["Nairobi"], ["Buenos", "Aires"], ["Osaka"], ["Oslo"], ["Montreal"], ["Cape", "Town"], ["Lima"],
        ["New", "Delhi"], ["Rotterdam"], ["Valparaiso"], ["Tbilisi"]]
_TEMPLATES = [
    "{PER} joined {ORG} in {LOC} last year .",
    "the team at {ORG} hired {PER} to lead the data platform .",
    "{PER} flew from {LOC} to {LOC} for the incident review .",
    "a breach at {ORG} was reported by {PER} on monday .",
    "{ORG} opened an office in {LOC} and moved {PER} there .",
    "according to {PER} , the model at {ORG} never worked in production .",
    "{PER} and {PER} presented the audit results in {LOC} .",
    "the {LOC} branch of {ORG} rolled back the deployment .",
    "nobody at {ORG} could explain the dashboard to {PER} .",
    "{PER} left {ORG} after the migration to {LOC} failed .",
]


def make_ner(n, seed=0, unseen_names=False, noise=0.15):
    """Sentences start with a capital letter, as in real text, and each entity token is lowercased with probability
    `noise` (chat logs, tickets, OCR). unseen_names=True draws people, organizations and places from the other half of
    each list: a test of generalization, not memory."""
    r = random.Random(seed)
    half = lambda xs: xs[len(xs) // 2:] if unseen_names else xs[:len(xs) // 2]
    first, last, orgs, locs = half(_FIRST), half(_LAST), half(_ORG), half(_LOC)
    data = []
    for _ in range(n):
        toks, tags = [], []
        for piece in r.choice(_TEMPLATES).split():
            if piece in ("{PER}", "{ORG}", "{LOC}"):
                kind = piece[1:-1]
                ent = [r.choice(first), r.choice(last)] if kind == "PER" else list(r.choice(orgs if kind == "ORG" else locs))
                if kind == "PER" and r.random() < 0.3:
                    ent = ent[1:]                                            # sometimes just the surname
                ent = [w.lower() if r.random() < noise else w for w in ent]
                toks += ent
                tags += [f"B-{kind}"] + [f"I-{kind}"] * (len(ent) - 1)
            else:
                toks.append(piece)
                tags.append("O")
        toks[0] = toks[0][0].upper() + toks[0][1:]
        data.append((toks, tags))
    return data
