"""Resolves the apparent tension with Zhao et al. 2024 (arXiv:2410.16090),
who report that a TRAINED linear probe detects knowledge-conflict PRESENCE
at intermediate layers (13th/17th of Llama3-8B) with high accuracy.

Our earlier tests found ~zero signal at the "mid" layer (12/36 Qwen) -- but
that was a different, harder, UNTRAINED question: "does the raw cosine
similarity between congruent and contradictory hidden states differ from
ordinary cross-entity variation." Zhao et al.'s task is easier and different:
"train a probe to classify conflict-present vs conflict-absent," which only
needs the signal to be LINEARLY DECODABLE, not visible in raw geometry.

This replicates their actual method on our data: train logistic regression
on mid-layer (12) and late-layer (30) hidden states to classify B (congruent,
label=0) vs C (contradictory, label=1), held out on unseen entities, across
the same 3 relations x 45 facts already used. If a trained mid-layer probe
generalizes well even though our untrained cosine check found nothing there,
that resolves the discrepancy as a method difference, not a real disagreement.

Run: python -m nous.replicate_zhao_conflict_probe
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from nous.verifier_multi_relation_robustness import RELATIONS

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYERS = [12, 30]
SEED = 0


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("mps")
    model.eval()
    captured = {}

    def make_hook(layer):
        def hook(_m, _i, out):
            captured[layer] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].float().clone()
        return hook

    for layer in LAYERS:
        model.model.layers[layer].register_forward_hook(make_hook(layer))
    return tok, model, captured


@torch.no_grad()
def feats_for(tok, model, captured, prompt: str) -> dict[int, np.ndarray]:
    ids = tok(prompt, return_tensors="pt").to("mps")
    model(**ids)
    return {layer: captured[layer].cpu().numpy() for layer in LAYERS}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def demo() -> None:
    """Self-check: hooks fire, congruent and contradictory prompts produce
    distinct feature vectors, before trusting the full 45-fact sweep."""
    tok, model, captured = load_model()
    fb = feats_for(tok, model, captured, "Document: France is a country whose capital is Paris. "
                                          "Q: What is the capital of France? A:")
    fc = feats_for(tok, model, captured, "Document: France is a country whose capital is Munich. "
                                          "Q: What is the capital of France? A:")
    for layer in LAYERS:
        assert not np.allclose(fb[layer], fc[layer])


def run() -> None:
    print("=" * 84)
    print(f"  Replicating Zhao et al.'s conflict-presence probe on {MODEL_ID}, layers {LAYERS}")
    print("=" * 84)

    tok, model, captured = load_model()
    rng = np.random.default_rng(SEED)

    feats = {layer: [] for layer in LAYERS}
    labels, group = [], []
    for rel_name, spec in RELATIONS.items():
        facts = spec["facts"]
        heads = [h for h, _ in facts]
        tails = {h: t for h, t in facts}
        false_tail = {}
        for h in heads:
            others = [t for hh, t in facts if hh != h]
            false_tail[h] = others[rng.integers(len(others))]

        for h in heads:
            prompt_b = f"Document: {spec['doc'].format(h=h, t=tails[h])} Q: {spec['q'].format(h=h)} A:"
            prompt_c = f"Document: {spec['doc'].format(h=h, t=false_tail[h])} Q: {spec['q'].format(h=h)} A:"
            fb = feats_for(tok, model, captured, prompt_b)
            fc = feats_for(tok, model, captured, prompt_c)
            labels.append(0); group.append((rel_name, h))
            labels.append(1); group.append((rel_name, h))
            for layer in LAYERS:
                feats[layer].append(fb[layer])
                feats[layer].append(fc[layer])

    labels = np.array(labels)
    n_pairs = len(labels) // 2
    perm = rng.permutation(n_pairs)
    train_idx_pairs, test_idx_pairs = perm[:int(n_pairs * 0.8)], perm[int(n_pairs * 0.8):]
    train_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    for p in train_idx_pairs:
        train_mask[2 * p] = train_mask[2 * p + 1] = True
    for p in test_idx_pairs:
        test_mask[2 * p] = test_mask[2 * p + 1] = True

    print(f"\n  n pairs = {n_pairs} (train {len(train_idx_pairs)}, held-out {len(test_idx_pairs)})")
    print(f"\n  {'layer':>6}{'held-out AUROC (trained probe)':>34}")
    for layer in LAYERS:
        X = np.stack(feats[layer])
        X_train = torch.tensor(X[train_mask], dtype=torch.float32)
        y_train = torch.tensor(labels[train_mask], dtype=torch.float32)
        X_test = torch.tensor(X[test_mask], dtype=torch.float32)
        y_test = labels[test_mask]

        mu, sigma = X_train.mean(0), X_train.std(0) + 1e-6
        probe = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        Xn = (X_train - mu) / sigma
        for _ in range(300):
            opt.zero_grad()
            loss = loss_fn(probe(Xn).squeeze(-1), y_train)
            loss.backward(); opt.step()

        with torch.no_grad():
            Xtn = (X_test - mu) / sigma
            scores = torch.sigmoid(probe(Xtn).squeeze(-1)).numpy()
        print(f"  {layer:>6}{auroc(scores, y_test):>34.3f}")


if __name__ == "__main__":
    demo()
    run()
