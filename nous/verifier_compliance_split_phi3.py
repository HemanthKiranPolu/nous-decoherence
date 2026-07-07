"""Does late-layer decoherence come from PROCESSING a contradiction, or from
COMMITTING to the false answer? verifier_cross_model_phi3.py only kept
per-relation-per-seed aggregates, so this rerun keeps the per-fact tuple
(relation, seed, head, overridden, decoherence) instead of collapsing it
immediately -- same forward passes as before, finer-grained bookkeeping.

Phi-3 is the right dataset for this because its override rate is much lower
than Qwen's (22-64% vs 71-98%), giving a naturally large, non-degenerate
"stubborn" bucket instead of Qwen's n=5.

Three profiles under test, comparing STUBBORN (model resisted, kept the true
answer) vs COMPLIANT (model followed the false document) vs BASELINE (ambient
cross-entity variation, no conflict at all -- same as the "cross" metric in
prior scripts):
  A. Pure conflict detector : stubborn ~= compliant >> baseline
  B. Active resistance      : stubborn > compliant >> baseline
  C. Active lie detector    : compliant > stubborn ~= baseline

Run: python -m nous.verifier_compliance_split_phi3
"""
from __future__ import annotations

import numpy as np

from nous.verifier_cross_model_phi3 import LAYERS, SEEDS, feats_for, load_model, short_completion
from nous.verifier_multi_relation_robustness import RELATIONS


def demo() -> None:
    """Self-check: model loads and the override premise still holds before
    trusting the full per-fact sweep (same check as verifier_cross_model_phi3)."""
    tok, model, captured = load_model()
    f = feats_for(tok, model, captured, "The capital of France is Paris.")
    for layer in LAYERS:
        assert f[layer].shape[0] == model.config.hidden_size
    completion = short_completion(
        tok, model, "Document: France is a country whose capital is Munich. "
                    "Q: What is the capital of France? A:")
    assert "Munich" in completion


def run() -> None:
    print("=" * 84)
    print("  Compliance split (Phi-3): does decoherence track conflict or commitment?")
    print("=" * 84)

    tok, model, captured = load_model()
    rows = []  # (relation, seed, head, overridden, decoherence[layer], is_baseline_pair)

    for rel_name, spec in RELATIONS.items():
        facts = spec["facts"]
        heads = [h for h, _ in facts]
        tails = {h: t for h, t in facts}

        b_feats = {}
        for h in heads:
            doc = spec["doc"].format(h=h, t=tails[h])
            prompt = f"Document: {doc} Q: {spec['q'].format(h=h)} A:"
            b_feats[h] = feats_for(tok, model, captured, prompt)

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            false_tail = {}
            for h in heads:
                others = [t for hh, t in facts if hh != h]
                false_tail[h] = others[rng.integers(len(others))]

            idx_i = rng.permutation(len(heads))
            idx_j = (idx_i + 1) % len(heads)
            baseline_pairs = {heads[i]: heads[j] for i, j in zip(idx_i, idx_j)}

            for h in heads:
                doc = spec["doc"].format(h=h, t=false_tail[h])
                prompt = f"Document: {doc} Q: {spec['q'].format(h=h)} A:"
                fc = feats_for(tok, model, captured, prompt)
                completion = short_completion(tok, model, prompt)
                overridden = int(false_tail[h] in completion and tails[h] not in completion)

                for layer in LAYERS:
                    B, C = b_feats[h][layer], fc[layer]
                    same_sim = float(np.dot(B, C) / (np.linalg.norm(B) * np.linalg.norm(C)))
                    other = b_feats[baseline_pairs[h]][layer]
                    base_sim = float(np.dot(B, other) / (np.linalg.norm(B) * np.linalg.norm(other)))
                    rows.append((rel_name, seed, h, layer, overridden, 1 - same_sim, 1 - base_sim))

    print(f"\n  collected {len(rows)} (relation, seed, head, layer) rows")

    for layer in LAYERS:
        layer_rows = [r for r in rows if r[3] == layer]
        stubborn = [r[5] for r in layer_rows if r[4] == 0]
        compliant = [r[5] for r in layer_rows if r[4] == 1]
        baseline = [r[6] for r in layer_rows]

        print(f"\n  --- layer {layer} ---")
        print(f"  n stubborn={len(stubborn)}  n compliant={len(compliant)}  n baseline={len(baseline)}")
        print(f"  mean decoherence: stubborn={np.mean(stubborn):.4f}  "
              f"compliant={np.mean(compliant):.4f}  baseline={np.mean(baseline):.4f}")

        s, c, b = np.mean(stubborn), np.mean(compliant), np.mean(baseline)
        if abs(s - c) < 0.15 * max(s, c) and min(s, c) > 1.3 * b:
            profile = "A: pure conflict detector (stubborn ~= compliant >> baseline)"
        elif s > c and c > 1.1 * b:
            profile = "B: active resistance (stubborn > compliant >> baseline)"
        elif c > s and abs(s - b) < 0.15 * max(s, b):
            profile = "C: active lie detector (compliant > stubborn ~= baseline)"
        else:
            profile = "none of A/B/C cleanly -- report raw numbers, don't force a label"
        print(f"  profile: {profile}")


if __name__ == "__main__":
    demo()
    run()
