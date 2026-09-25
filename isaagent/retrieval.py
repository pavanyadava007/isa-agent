"""Manual retrieval: BM25 over manual chunks, exact lookup by operation name, fuzzy name suggestions."""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter

from isaagent.gen_isa import manual_entries

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        if t.startswith("xd") and len(t) > 3:   # xd_hqmulx -> also "hqmulx" pieces are in the token already
            out.append(t[2:])
    return out


class Manual:
    def __init__(self) -> None:
        self.chunks = manual_entries()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.op_names = [c["id"] for c in self.chunks if not c["id"].startswith("chapter:")]
        self.docs = [tokenize(c["title"] + " " + c["text"]) for c in self.chunks]
        self.df = Counter(t for d in self.docs for t in set(d))
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)

    def bm25(self, query: str, k: int = 8, k1: float = 1.2, b: float = 0.75, ops_only: bool = True) -> list[dict]:
        q = tokenize(query)
        n = len(self.docs)
        scores = []
        for i, d in enumerate(self.docs):
            if ops_only and self.chunks[i]["id"].startswith("chapter:"):
                continue
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for s, i in scores[:k] if s > 0]

    def lookup(self, name: str) -> dict | None:
        return self.by_id.get(name)

    def suggest(self, name: str, k: int = 4, semantic: bool = True) -> list[str]:
        """Closest real operations for an invented name: spelling neighbours plus (semantic=True) a manual
        search on the words inside the name, e.g. xd_wzero -> 'w zero' -> xd_wdup ("To zero an accumulator...")."""
        close = difflib.get_close_matches(name, self.op_names, n=k, cutoff=0.5)
        if not semantic:
            return close
        words = re.sub(r"^xd_", "", name)
        words = " ".join([words[:1], words[1:]] + words.split("_"))
        sem = [c["id"] for c in self.bm25(words, k=k)]
        out = list(dict.fromkeys(sem[:2] + close + sem[2:]))
        return out[:k + 1]

    def chapters(self, names: list[str] | None = None) -> list[dict]:
        cs = [c for c in self.chunks if c["id"].startswith("chapter:")]
        if names is not None:
            cs = [c for c in cs if c["title"] in names]
        return cs

    def full(self) -> list[dict]:
        return list(self.chunks)


def render(chunks: list[dict]) -> str:
    seen, out = set(), []
    for c in chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(f"### {c['title']}\n{c['text']}")
    return "\n\n".join(out)
