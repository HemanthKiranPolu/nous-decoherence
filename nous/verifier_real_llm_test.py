"""Stage 1 verifier (verifier.py), re-tested on REAL LLM hidden states instead
of simulated hallucinations -- the honest scope note in NOUS_LLM_ARCHITECTURE.md
flagged this as the next step ("hallucinations here are simulated... real-LLM
validation is the next step"). This is that step.

Claim under test: does a local GPT-2's internal hidden state, for a sentence
asserting a fact, carry a linearly-recoverable signal of whether the fact is
TRUE or a plausible HALLUCINATION -- generalizing to countries the probe never
saw, not just memorizing the training sentences?

Design (country/capital facts -- real, unambiguous, well inside GPT-2's 2019
WebText training window, so any signal found is prior pretrained knowledge, not
something we taught it):
  - 30 country->capital facts. TRUE sentence: "The capital of X is Y."
    FALSE (plausible hard negative, not gibberish): "The capital of X is Z."
    where Z is another country's REAL capital (type-consistent, exactly the
    hallucination shape verifier.py already tests symbolically).
  - Register a forward hook on a middle GPT-2 layer, capture the LAST-token
    hidden state (768-dim) for each sentence.
  - Split countries 20 train / 10 held-out. Train a LINEAR probe (logistic
    regression) hidden-state -> P(true) on the 20 train countries' true+false
    sentences only. Evaluate AUROC on the 10 held-out countries' sentences --
    the probe never saw these countries' hidden states during training.
  - Separately, report NOUS's own symbolic AUROC on the full fact set (as
    verifier.py does) as a reference point -- NOT a fair apples-to-apples split
    against the probe: NOUS is a bijective single-relation lookup here (one
    capital per country), so it has no meaningful held-out-FACT axis the way
    UMLS's many-to-many relations do; its number reflects "does its confidence
    separate true/false among facts it was trained on," while the probe number
    reflects genuine generalization to unseen countries. Both are real, they're
    just not measuring the same thing, and conflating them would overclaim.

Run: python -m nous.verifier_real_llm_test
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from nous.relational import NOUSRelational

FACTS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Egypt", "Cairo"), ("Brazil", "Brasilia"), ("Mexico", "Mexico City"),
    ("India", "New Delhi"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Turkey", "Ankara"), ("Thailand", "Bangkok"),
    ("Argentina", "Buenos Aires"), ("Kenya", "Nairobi"), ("Ireland", "Dublin"), ("Austria", "Vienna"),
    ("Belgium", "Brussels"), ("Switzerland", "Bern"), ("Netherlands", "Amsterdam"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Vietnam", "Hanoi"),
]
LAYER = 6
SEED = 0


def load_gpt2():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=False)
    model.eval()
    captured = {}

    def hook(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    model.transformer.h[LAYER].register_forward_hook(hook)
    return tok, model, captured


@torch.no_grad()
def hidden_state(tok, model, captured, sentence: str) -> torch.Tensor:
    ids = tok(sentence, return_tensors="pt")
    model(**ids)
    return captured["h"][0, -1, :].clone()  # last token, chosen layer


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def demo() -> None:
    """Self-check: hook captures a 768-dim vector, distinct sentences -> distinct vectors."""
    tok, model, captured = load_gpt2()
    v1 = hidden_state(tok, model, captured, "The capital of France is Paris.")
    v2 = hidden_state(tok, model, captured, "The capital of France is Berlin.")
    assert v1.shape == (768,)
    assert not torch.allclose(v1, v2)


def run() -> None:
    print("=" * 78)
    print("  Verifier on REAL LLM hidden states (GPT-2) vs NOUS symbolic verifier")
    print("=" * 78)

    rng = np.random.default_rng(SEED)
    countries = [c for c, _ in FACTS]
    capitals = {c: cap for c, cap in FACTS}
    perm = rng.permutation(len(countries))
    train_countries = [countries[i] for i in perm[:20]]
    test_countries = [countries[i] for i in perm[20:]]

    def false_capital(country: str) -> str:
        others = [cap for c, cap in FACTS if c != country]
        return others[rng.integers(len(others))]

    print("\n--- capturing GPT-2 hidden states (layer 6, last token) ---")
    tok, model, captured = load_gpt2()
    feats, labels, group = [], [], []
    for country in countries:
        true_sent = f"The capital of {country} is {capitals[country]}."
        false_sent = f"The capital of {country} is {false_capital(country)}."
        feats.append(hidden_state(tok, model, captured, true_sent).numpy()); labels.append(1); group.append(country)
        feats.append(hidden_state(tok, model, captured, false_sent).numpy()); labels.append(0); group.append(country)
    feats = np.stack(feats); labels = np.array(labels); group = np.array(group)

    train_mask = np.isin(group, train_countries)
    test_mask = np.isin(group, test_countries)

    print(f"  train countries: {len(train_countries)}   held-out countries: {len(test_countries)}")

    print("\n--- training linear probe on TRAIN countries only ---")
    X_train = torch.tensor(feats[train_mask], dtype=torch.float32)
    y_train = torch.tensor(labels[train_mask], dtype=torch.float32)
    X_test = torch.tensor(feats[test_mask], dtype=torch.float32)
    y_test = labels[test_mask]

    mu, sigma = X_train.mean(0), X_train.std(0) + 1e-6
    X_train_n, X_test_n = (X_train - mu) / sigma, (X_test - mu) / sigma

    probe = nn.Linear(768, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(probe(X_train_n).squeeze(-1), y_train)
        loss.backward(); opt.step()

    with torch.no_grad():
        probe_scores = torch.sigmoid(probe(X_test_n).squeeze(-1)).numpy()
    probe_auroc = auroc(probe_scores, y_test)
    print(f"  held-out AUROC (probe on unseen countries' hidden states): {probe_auroc:.3f}")

    print("\n--- reference: NOUS symbolic verifier on the same fact set ---")
    print("  (in-domain check, not the same held-out axis -- see module docstring)")
    all_entities = sorted({c for c, _ in FACTS} | {cap for _, cap in FACTS})
    eid = {e: i for i, e in enumerate(all_entities)}
    nous = NOUSRelational(len(all_entities), 1, dim=32, asymmetry=0.0)
    TR = torch.tensor([[eid[c], 0, eid[cap]] for c, cap in FACTS])
    nous.fit(TR, epochs=200, log_every=0)

    nous_scores, nous_labels = [], []
    with torch.no_grad():
        for country in countries:
            s = torch.softmax(nous.score_tail(torch.tensor([eid[country]]), torch.tensor([0]))[0], -1)
            nous_scores.append(s[eid[capitals[country]]].item()); nous_labels.append(1)
            fc = false_capital(country)
            nous_scores.append(s[eid[fc]].item()); nous_labels.append(0)
    nous_auroc = auroc(np.array(nous_scores), np.array(nous_labels))
    print(f"  NOUS AUROC (symbolic, saw all facts during training): {nous_auroc:.3f}")

    print("\n" + "=" * 78)
    print(f"  SUMMARY: real-hidden-state probe (held-out countries) = {probe_auroc:.3f}"
          f"   |   NOUS symbolic (in-domain) = {nous_auroc:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    demo()
    run()
