"""Does a truth probe trained on PARAMETRIC recall still detect the
pretrained-true fact in a model's hidden state when in-context evidence has
made it OUTPUT the false one?

verifier_real_llm_test.py validated that a linear probe on GPT-2 hidden states
recovers true/false parametric claims, generalizing to unseen entities (AUROC
0.980). That test never put the model under any pressure to say something
false -- there was no context arguing for the wrong answer. This is the harder
case pitched as the actual RAG-verifier scenario: a document CONTRADICTS the
model's parametric prior, and the model follows the document.

Premise check first (matters before building anything else): GPT-2-base does
NOT reliably follow an in-context override (checked directly -- "Munich"
doesn't even reach its top-5 completion for a contradictory document, and it
doesn't even nail the congruent case correctly). Switched to Qwen3-4B-Instruct
(instruction-tuned, already cached locally), which does override correctly:
"The capital of France is" -> Paris; congruent-document QA -> Paris;
contradictory-document QA -> Munich. That override is the whole premise of
this test, so it's checked at scale (all 30 countries) below, not just on one
cherry-picked example.

Three prompt types per country:
  A parametric : "The capital of {country} is"                 (bare recall)
  B congruent  : "Document: ... capital is {TRUE}. Q: ... A:"  (context agrees)
  C contradict : "Document: ... capital is {FALSE}. Q: ... A:" (context lies)

Probe: trained ONLY on declarative true/false sentences (Type-A style, no
context, same recipe as verifier_real_llm_test.py) on 20 countries, at two
layers (mid=12, late=30 of 36). Then applied FROZEN to Type B and Type C
hidden states for all 30 countries, to see whether it still registers the
pretrained-true answer's signature when the model is about to output the
context-supplied false one.

Run: python -m nous.verifier_in_context_conflict
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYERS = [12, 30]  # mid / late, of 36
SEED = 0

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
def run_prompt(tok, model, captured, prompt: str):
    ids = tok(prompt, return_tensors="pt").to("mps")
    logits = model(**ids).logits[0, -1]
    top1 = tok.decode([logits.argmax().item()]).strip()
    return top1, {layer: captured[layer].cpu().numpy() for layer in LAYERS}


@torch.no_grad()
def short_completion(tok, model, prompt: str, max_new_tokens: int = 12) -> str:
    """Many answers preface the capital with filler ('The document states...')
    before naming it -- checking only the literal first token undercounts
    overrides that are real but delayed by a few tokens. Generate a short
    continuation instead and search the whole thing."""
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def demo() -> None:
    """Self-check: model loads, hooks fire on both layers, override happens on
    the canonical France/Munich example before trusting the full run."""
    tok, model, captured = load_model()
    top1, feats = run_prompt(tok, model, captured, "The capital of France is")
    assert "Paris" in top1
    for layer in LAYERS:
        assert feats[layer].shape[0] == model.config.hidden_size
    top1_c, _ = run_prompt(
        tok, model, captured,
        "Document: France is a country whose capital is Munich. Q: What is the capital of France? A:")
    assert "Munich" in top1_c


def run() -> None:
    print("=" * 78)
    print(f"  In-context conflict test on {MODEL_ID}, layers {LAYERS}")
    print("=" * 78)

    rng = np.random.default_rng(SEED)
    countries = [c for c, _ in FACTS]
    capitals = {c: cap for c, cap in FACTS}

    def false_capital(country: str) -> str:
        others = [cap for c, cap in FACTS if c != country]
        return others[rng.integers(len(others))]

    tok, model, captured = load_model()

    print("\n--- premise check: does the model actually override, across all 30 countries? ---")
    print("  (checked via a short generated continuation, not just the literal first token --")
    print("   many answers preface the capital with filler before naming it)")
    overrides, total = 0, 0
    false_cap_for = {c: false_capital(c) for c in countries}
    for country in countries:
        fc = false_cap_for[country]
        prompt = (f"Document: {country} is a country whose capital is {fc}. "
                  f"Q: What is the capital of {country}? A:")
        completion = short_completion(tok, model, prompt)
        total += 1
        followed = fc in completion and capitals[country] not in completion
        overrides += int(followed)
    print(f"  model followed the contradictory document: {overrides}/{total} countries")

    perm = rng.permutation(len(countries))
    train_countries = [countries[i] for i in perm[:20]]
    test_countries = [countries[i] for i in perm[20:]]

    print("\n--- training parametric truth probes (declarative sentences, 20 countries) ---")
    feats_by_layer = {layer: [] for layer in LAYERS}
    labels, group = [], []
    for country in countries:
        true_sent = f"The capital of {country} is {capitals[country]}."
        false_sent = f"The capital of {country} is {false_cap_for[country]}."
        _, f_true = run_prompt(tok, model, captured, true_sent)
        _, f_false = run_prompt(tok, model, captured, false_sent)
        labels.append(1); group.append(country)
        labels.append(0); group.append(country)
        for layer in LAYERS:
            feats_by_layer[layer].append(f_true[layer])
            feats_by_layer[layer].append(f_false[layer])
    labels = np.array(labels); group = np.array(group)
    train_mask = np.isin(group, train_countries)
    test_mask = np.isin(group, test_countries)

    probes = {}
    for layer in LAYERS:
        X = np.stack(feats_by_layer[layer])
        X_train = torch.tensor(X[train_mask], dtype=torch.float32)
        y_train = torch.tensor(labels[train_mask], dtype=torch.float32)
        mu, sigma = X_train.mean(0), X_train.std(0) + 1e-6
        probe = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        Xn = (X_train - mu) / sigma
        for _ in range(300):
            opt.zero_grad()
            loss = loss_fn(probe(Xn).squeeze(-1), y_train)
            loss.backward(); opt.step()
        probes[layer] = (probe, mu, sigma)

        X_test = torch.tensor((X[test_mask] - mu.numpy()) / sigma.numpy(), dtype=torch.float32)
        with torch.no_grad():
            held_out_scores = torch.sigmoid(probe(X_test).squeeze(-1)).numpy()
        print(f"  layer {layer}: held-out declarative AUROC (10 unseen countries) = "
              f"{auroc(held_out_scores, labels[test_mask]):.3f}")

    print("\n--- applying the FROZEN probe to Type B (congruent) / Type C (contradictory) QA prompts ---")
    print("  (probe was trained on short declarative sentences -- format mismatch vs these longer")
    print("   QA prompts is a real confound, watch for saturated ~1.0 scores on both types)")
    print(f"\n  {'layer':>6}{'type B: P(true) mean':>24}{'type C: P(true) mean':>24}")
    b_feats, c_feats = {layer: [] for layer in LAYERS}, {layer: [] for layer in LAYERS}
    for layer in LAYERS:
        probe, mu, sigma = probes[layer]
        b_scores, c_scores = [], []
        for country in countries:
            fc = false_cap_for[country]
            prompt_b = (f"Document: {country} is a country whose capital is {capitals[country]}. "
                        f"Q: What is the capital of {country}? A:")
            prompt_c = (f"Document: {country} is a country whose capital is {fc}. "
                        f"Q: What is the capital of {country}? A:")
            _, fb = run_prompt(tok, model, captured, prompt_b)
            _, fc_feat = run_prompt(tok, model, captured, prompt_c)
            b_feats[layer].append(fb[layer]); c_feats[layer].append(fc_feat[layer])
            with torch.no_grad():
                xb = (torch.tensor(fb[layer], dtype=torch.float32) - mu) / sigma
                xc = (torch.tensor(fc_feat[layer], dtype=torch.float32) - mu) / sigma
                b_scores.append(torch.sigmoid(probe(xb)).item())
                c_scores.append(torch.sigmoid(probe(xc)).item())
        print(f"  {layer:>6}{np.mean(b_scores):>24.3f}{np.mean(c_scores):>24.3f}")

    print("\n--- cosine-similarity decoherence check (same format both sides, no probe confound) ---")
    print("  same-country B-vs-C (congruent vs contradictory, only the capital differs)")
    print("  compared against cross-country B-vs-B (different content, same 'congruent' format)")
    rng2 = np.random.default_rng(SEED + 1)
    print(f"\n  {'layer':>6}{'same-country B vs C':>22}{'cross-country B vs B':>22}")
    for layer in LAYERS:
        B = np.stack(b_feats[layer]); C = np.stack(c_feats[layer])
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        same_country = (Bn * Cn).sum(1)
        idx_i = rng2.permutation(len(countries))
        idx_j = (idx_i + 1) % len(countries)
        cross_country = (Bn[idx_i] * Bn[idx_j]).sum(1)
        print(f"  {layer:>6}{same_country.mean():>22.4f}{cross_country.mean():>22.4f}")


if __name__ == "__main__":
    demo()
    run()
