"""Robustness check for verifier_in_context_conflict.py's one finding: at
layer 30, a same-entity in-context contradiction shifted Qwen3-4B-Instruct's
hidden state MORE than switching to a different country entirely (0.928 vs
0.983 cosine similarity). That was one relation (capital_of), one seed. Before
calling it a "physical constant of transformer conflict," the obvious next
check is whether the same gap shows up on relations the model wasn't asked
about before, across multiple random seeds -- not whether it sounds right.

Three relations, deliberately different from each other and from capital_of:
  capital_of  : country -> capital        (repeat, as the known-working case)
  founded_by  : company -> founder        (people, not places)
  written_by  : book -> author            (titles, not proper-noun entities)

For each relation x seed: confirm the model actually overrides (same premise
check as before, via short generated completions, not first-token-only), then
compute the same same-entity-B-vs-C vs cross-entity-B-vs-B cosine comparison
at layers 12 and 30.

Run: python -m nous.verifier_multi_relation_robustness
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYERS = [12, 30]
SEEDS = [0, 1, 2]

RELATIONS = {
    "capital_of": {
        "doc": "{h} is a country whose capital is {t}.",
        "q": "What is the capital of {h}?",
        "facts": [
            ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Japan", "Tokyo"),
            ("China", "Beijing"), ("Russia", "Moscow"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
            ("Brazil", "Brasilia"), ("India", "New Delhi"), ("Greece", "Athens"), ("Poland", "Warsaw"),
            ("Sweden", "Stockholm"), ("Turkey", "Ankara"), ("Argentina", "Buenos Aires"),
        ],
    },
    "founded_by": {
        "doc": "{h} is a company founded by {t}.",
        "q": "Who founded {h}?",
        "facts": [
            ("Microsoft", "Bill Gates"), ("Facebook", "Mark Zuckerberg"), ("Amazon", "Jeff Bezos"),
            ("Tesla", "Elon Musk"), ("Apple", "Steve Jobs"), ("Dell", "Michael Dell"),
            ("Oracle", "Larry Ellison"), ("Nike", "Phil Knight"), ("IKEA", "Ingvar Kamprad"),
            ("Ford", "Henry Ford"), ("Disney", "Walt Disney"), ("Netflix", "Reed Hastings"),
            ("Twitter", "Jack Dorsey"), ("LinkedIn", "Reid Hoffman"), ("Airbnb", "Brian Chesky"),
        ],
    },
    "written_by": {
        "doc": "{h} is a book written by {t}.",
        "q": "Who wrote {h}?",
        "facts": [
            ("1984", "George Orwell"), ("Hamlet", "William Shakespeare"), ("Dracula", "Bram Stoker"),
            ("Frankenstein", "Mary Shelley"), ("Moby Dick", "Herman Melville"),
            ("Pride and Prejudice", "Jane Austen"), ("War and Peace", "Leo Tolstoy"),
            ("The Odyssey", "Homer"), ("Don Quixote", "Miguel de Cervantes"),
            ("Crime and Punishment", "Fyodor Dostoevsky"), ("The Great Gatsby", "F. Scott Fitzgerald"),
            ("Great Expectations", "Charles Dickens"), ("Wuthering Heights", "Emily Bronte"),
            ("The Hobbit", "J.R.R. Tolkien"), ("Brave New World", "Aldous Huxley"),
        ],
    },
}


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


@torch.no_grad()
def short_completion(tok, model, prompt: str, max_new_tokens: int = 14) -> str:
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def demo() -> None:
    """Self-check: model loads, hooks fire, override happens on one known-good
    relation/example before trusting the full multi-relation sweep."""
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
    print(f"  Multi-relation robustness check: {MODEL_ID}, layers {LAYERS}, seeds {SEEDS}")
    print("=" * 84)

    tok, model, captured = load_model()

    print(f"\n  {'relation':<14}{'seed':>6}{'override':>12}"
          f"{'L12 same':>11}{'L12 cross':>11}{'L30 same':>11}{'L30 cross':>11}")
    summary = {rel: {layer: [] for layer in LAYERS} for rel in RELATIONS}

    for rel_name, spec in RELATIONS.items():
        facts = spec["facts"]
        heads = [h for h, _ in facts]
        tails = {h: t for h, t in facts}

        b_feats = {layer: [] for layer in LAYERS}
        for h in heads:
            doc = spec["doc"].format(h=h, t=tails[h])
            prompt = f"Document: {doc} Q: {spec['q'].format(h=h)} A:"
            f = feats_for(tok, model, captured, prompt)
            for layer in LAYERS:
                b_feats[layer].append(f[layer])

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            false_tail = {}
            for h in heads:
                others = [t for hh, t in facts if hh != h]
                false_tail[h] = others[rng.integers(len(others))]

            c_feats = {layer: [] for layer in LAYERS}
            overrides = 0
            for h in heads:
                doc = spec["doc"].format(h=h, t=false_tail[h])
                prompt = f"Document: {doc} Q: {spec['q'].format(h=h)} A:"
                f = feats_for(tok, model, captured, prompt)
                for layer in LAYERS:
                    c_feats[layer].append(f[layer])
                completion = short_completion(tok, model, prompt)
                if false_tail[h] in completion and tails[h] not in completion:
                    overrides += 1

            row = {"override": f"{overrides}/{len(heads)}"}
            for layer in LAYERS:
                B = np.stack(b_feats[layer]); C = np.stack(c_feats[layer])
                Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
                Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
                same = (Bn * Cn).sum(1).mean()
                idx_i = rng.permutation(len(heads))
                idx_j = (idx_i + 1) % len(heads)
                cross = (Bn[idx_i] * Bn[idx_j]).sum(1).mean()
                row[layer] = (same, cross)
                summary[rel_name][layer].append((same, cross, overrides / len(heads)))

            print(f"  {rel_name:<14}{seed:>6}{row['override']:>12}"
                  f"{row[12][0]:>11.4f}{row[12][1]:>11.4f}"
                  f"{row[30][0]:>11.4f}{row[30][1]:>11.4f}")

    print("\n" + "=" * 84)
    print("  SUMMARY (mean over seeds)")
    print("=" * 84)
    print(f"\n  {'relation':<14}{'override %':>12}{'L12 gap (cross-same)':>24}{'L30 gap (cross-same)':>24}")
    for rel_name in RELATIONS:
        ov = np.mean([o for _, _, o in summary[rel_name][LAYERS[0]]]) * 100
        gaps = {}
        for layer in LAYERS:
            vals = summary[rel_name][layer]
            gap = np.mean([c - s for s, c, _ in vals])
            gaps[layer] = gap
        print(f"  {rel_name:<14}{ov:>11.0f}%{gaps[12]:>24.4f}{gaps[30]:>24.4f}")


if __name__ == "__main__":
    demo()
    run()
