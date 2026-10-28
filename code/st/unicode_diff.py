"""Every code point, through our basic tokenizer and through the one sentence-transformers really uses."""
import unicodedata
from collections import Counter

from sentence_transformers import SentenceTransformer

from mini_st import basic_tokenize

backend = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").tokenizer.backend_tokenizer
groups, tested = Counter(), 0
for cp in range(0x20, 0x30000):
    if 0xD800 <= cp <= 0xDFFF:                        # surrogates aren't characters
        continue
    tested += 1
    text = "a" + chr(cp) + "b"
    theirs = [w for w, _ in backend.pre_tokenizer.pre_tokenize_str(backend.normalizer.normalize_str(text))]
    if basic_tokenize(text) != theirs:
        cat = unicodedata.category(chr(cp))
        if cat == "Cn":
            groups["unassigned in Unicode " + unicodedata.unidata_version] += 1
        elif 0x2B820 <= cp <= 0x2B91F:
            groups["CJK U+2B820 to U+2B91F"] += 1
        else:
            groups["other (category " + cat + ")"] += 1
print(f"{sum(groups.values())} of {tested} code points disagree")
for name, n in groups.most_common():
    print(f"{n:6d}  {name}")
