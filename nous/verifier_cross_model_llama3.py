"""Third model family for the layer-30-decoherence robustness check --
Llama-3.2-3B-Instruct itself, now that license access is available (unlike
verifier_cross_model_phi3.py, which had to substitute Phi-3 because Llama-3
was gated with no token at the time). Requires HF_TOKEN in the environment
with an accepted license for meta-llama/Llama-3.2-3B-Instruct.

Premise checked at scale first (not assumed): a single France/Munich example
showed Llama-3 correcting the document ("Munich is not the capital of
France... Paris"), which looked like a premise failure similar to GPT-2/
TinyLlama -- but checking across all 45 facts (3 relations x 15) shows real,
non-negligible override rates on every relation (capital_of 13/15, founded_by
3/15, written_by 4/15), so the single example was atypical, not representative,
and this model is usable.

Layers chosen by matching proportional depth to Qwen's 12/36 (~33%) and 30/36
(~83%): Llama-3.2-3B has 28 layers, so mid=9 (32%), late=23 (82%).

Run: HF_TOKEN=... python -m nous.verifier_cross_model_llama3
"""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nous.verifier_multi_relation_robustness import RELATIONS

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LAYERS = [9, 23]  # mid / late, of 28 -- proportionally matched to Qwen's 12/36 and 30/36
SEEDS = [0, 1, 2]


def load_model():
    token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, token=token).to("mps")
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
def short_completion(tok, model, prompt: str, max_new_tokens: int = 16) -> str:
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def demo() -> None:
    """Self-check: hooks fire on both layers before trusting the full sweep.
    Override is NOT asserted here on a single example -- checked directly and
    found unreliable per-example (see module docstring); the real check is the
    aggregate override rate printed in run()."""
    tok, model, captured = load_model()
    f = feats_for(tok, model, captured, "The capital of France is Paris.")
    for layer in LAYERS:
        assert f[layer].shape[0] == model.config.hidden_size


def run() -> None:
    print("=" * 84)
    print(f"  Cross-model check: {MODEL_ID}, layers {LAYERS}, seeds {SEEDS}")
    print("=" * 84)

    tok, model, captured = load_model()

    print(f"\n  {'relation':<14}{'seed':>6}{'override':>12}"
          f"{'L9 same':>10}{'L9 cross':>10}{'L23 same':>11}{'L23 cross':>11}")
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
                  f"{row[LAYERS[0]][0]:>10.4f}{row[LAYERS[0]][1]:>10.4f}"
                  f"{row[LAYERS[1]][0]:>11.4f}{row[LAYERS[1]][1]:>11.4f}")

    print("\n" + "=" * 84)
    print("  SUMMARY (mean over seeds)")
    print("=" * 84)
    print(f"\n  {'relation':<14}{'override %':>12}{'L9 gap (cross-same)':>23}{'L23 gap (cross-same)':>24}")
    for rel_name in RELATIONS:
        ov = np.mean([o for _, _, o in summary[rel_name][LAYERS[0]]]) * 100
        gaps = {}
        for layer in LAYERS:
            vals = summary[rel_name][layer]
            gaps[layer] = np.mean([c - s for s, c, _ in vals])
        print(f"  {rel_name:<14}{ov:>11.0f}%{gaps[LAYERS[0]]:>23.4f}{gaps[LAYERS[1]]:>24.4f}")


if __name__ == "__main__":
    demo()
    run()
