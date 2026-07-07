"""Second model family for the layer-30-decoherence robustness check.
verifier_multi_relation_robustness.py only ran Qwen3-4B-Instruct -- the open
question was whether the mid-layer/late-layer split is a Qwen quirk or a more
general transformer property. Llama-3 itself is gated (manual HF approval,
no token available here) -- checked directly, not assumed. TinyLlama-1.1B-Chat
(genuine Llama architecture) was tried first and failed the override premise
outright: given a document saying "France's capital is Munich," it still
answers "Paris" -- too weak to use context over parametric prior, same failure
mode GPT-2 had. Phi-3-mini-4k-instruct (Microsoft, different architecture team
than Qwen, ungated) passed the premise check (correctly outputs "Munich") and
is the model used here.

Layers chosen by matching proportional depth to Qwen's 12/36 (~33%) and 30/36
(~83%): Phi-3 has 32 layers, so mid=11 (34%), late=27 (84%).

Run: python -m nous.verifier_cross_model_phi3
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nous.verifier_multi_relation_robustness import RELATIONS

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LAYERS = [11, 27]  # mid / late, of 32 -- proportionally matched to Qwen's 12/36 and 30/36
SEEDS = [0, 1, 2]


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
    """Self-check: hooks fire on both layers, and the model actually overrides
    on the canonical example before trusting the full sweep."""
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
    print(f"  Cross-model check: {MODEL_ID}, layers {LAYERS}, seeds {SEEDS}")
    print("=" * 84)

    tok, model, captured = load_model()

    print(f"\n  {'relation':<14}{'seed':>6}{'override':>12}"
          f"{'L11 same':>11}{'L11 cross':>11}{'L27 same':>11}{'L27 cross':>11}")
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
                  f"{row[LAYERS[0]][0]:>11.4f}{row[LAYERS[0]][1]:>11.4f}"
                  f"{row[LAYERS[1]][0]:>11.4f}{row[LAYERS[1]][1]:>11.4f}")

    print("\n" + "=" * 84)
    print("  SUMMARY (mean over seeds)")
    print("=" * 84)
    print(f"\n  {'relation':<14}{'override %':>12}{'L11 gap (cross-same)':>24}{'L27 gap (cross-same)':>24}")
    for rel_name in RELATIONS:
        ov = np.mean([o for _, _, o in summary[rel_name][LAYERS[0]]]) * 100
        gaps = {}
        for layer in LAYERS:
            vals = summary[rel_name][layer]
            gap = np.mean([c - s for s, c, _ in vals])
            gaps[layer] = gap
        print(f"  {rel_name:<14}{ov:>11.0f}%{gaps[LAYERS[0]]:>24.4f}{gaps[LAYERS[1]]:>24.4f}")


if __name__ == "__main__":
    demo()
    run()
