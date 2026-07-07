"""Builds realistic, multi-sentence "RAG documents" on top of relations_large's
120-fact set -- the target fact is embedded among 2 other TRUE facts from the
same relation (real noise, not the single isolated sentence the earlier tests
used), with generic opening/closing lines. Congruent and contradictory
versions differ only in the target sentence's tail; everything else is
identical, so any decoherence measured isn't just "document got longer."
"""
from __future__ import annotations

import numpy as np

OPENERS = {
    "capital_of": "The following is an excerpt from a geography reference guide.",
    "founded_by": "The following is an excerpt from a business history archive.",
    "written_by": "The following is an excerpt from a literary reference guide.",
    "painted_by": "The following is an excerpt from an art history archive.",
    "directed_by": "The following is an excerpt from a film history archive.",
    "composed_by": "The following is an excerpt from a music history archive.",
}
CLOSERS = {
    "capital_of": "These are commonly referenced facts in world geography.",
    "founded_by": "These are commonly referenced facts in business history.",
    "written_by": "These are commonly referenced facts in literary history.",
    "painted_by": "These are commonly referenced facts in art history.",
    "directed_by": "These are commonly referenced facts in film history.",
    "composed_by": "These are commonly referenced facts in music history.",
}


def pick_noise_sentences(rel_name: str, spec: dict, head: str, rng: np.random.Generator) -> list[str]:
    """Chooses the 2 filler facts ONCE per head -- caller reuses the same
    choice for both the congruent and contradictory document so the two
    versions differ ONLY in the target sentence's tail, not in which noise
    facts happened to get drawn (calling rng.choice twice, once per document,
    would silently let the two documents diverge on filler content too)."""
    facts = spec["facts"]
    others = [(h, t) for h, t in facts if h != head]
    noise_idx = rng.choice(len(others), size=min(2, len(others)), replace=False)
    return [spec["doc"].format(h=others[i][0], t=others[i][1]) for i in noise_idx]


def build_document(rel_name: str, spec: dict, head: str, tail: str,
                    noise_sentences: list[str]) -> str:
    """Target fact embedded among the given (pre-chosen, shared-across-B/C)
    noise sentences, with generic opening/closing lines -- a realistic
    multi-sentence document instead of one isolated sentence."""
    target_sentence = spec["doc"].format(h=head, t=tail)
    body = noise_sentences[:1] + [target_sentence] + noise_sentences[1:]
    return " ".join([OPENERS[rel_name]] + body + [CLOSERS[rel_name]])
