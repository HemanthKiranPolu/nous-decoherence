"""Closes two of the three gaps flagged after publishing: small n (45 facts,
3 relations) and synthetic single-sentence-only documents. Uses
RELATIONS_LARGE (120 facts, 6 relations, relations_large.py) and realistic
multi-sentence documents (relations_large_docs.py, target fact embedded among
2 other true facts + generic opening/closing, not an isolated sentence) --
same decoherence + override-compliance measurement as
verifier_multi_relation_robustness.py, at scale, on real documents.

Run: python -m nous.verifier_scaled_realistic
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nous.relations_large import RELATIONS_LARGE
from nous.relations_large_docs import build_document, pick_noise_sentences

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYERS = [12, 30]
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
def short_completion(tok, model, prompt: str, max_new_tokens: int = 16) -> str:
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def demo() -> None:
    """Self-check: a built document embeds the target fact + 2 real noise
    facts + opener/closer, and the model still overrides on the canonical
    example when the target fact is buried in a longer, realistic document."""
    rng = np.random.default_rng(0)
    spec = RELATIONS_LARGE["capital_of"]
    noise = pick_noise_sentences("capital_of", spec, "France", rng)
    doc_b = build_document("capital_of", spec, "France", "Paris", noise)
    doc = build_document("capital_of", spec, "France", "Munich", noise)
    assert "France" in doc and "Munich" in doc
    assert doc.count(".") >= 4  # opener + >=2 facts + closer, at least
    # same noise sentences on both sides -- only the target tail differs
    assert doc.replace("Munich", "Paris") == doc_b

    tok, model, captured = load_model()
    prompt = f"Document: {doc} Q: What is the capital of France? A:"
    completion = short_completion(tok, model, prompt)
    assert "Munich" in completion


def run() -> None:
    print("=" * 88)
    print(f"  Scaled + realistic-document check: {MODEL_ID}, layers {LAYERS}, seeds {SEEDS}")
    print(f"  {sum(len(s['facts']) for s in RELATIONS_LARGE.values())} facts across "
          f"{len(RELATIONS_LARGE)} relations, multi-sentence documents (not single sentences)")
    print("=" * 88)

    tok, model, captured = load_model()

    print(f"\n  {'relation':<14}{'seed':>6}{'override':>12}"
          f"{'L12 same':>11}{'L12 cross':>11}{'L30 same':>11}{'L30 cross':>11}")
    summary = {rel: {layer: [] for layer in LAYERS} for rel in RELATIONS_LARGE}

    for rel_name, spec in RELATIONS_LARGE.items():
        facts = spec["facts"]
        heads = [h for h, _ in facts]
        tails = {h: t for h, t in facts}

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            false_tail = {}
            for h in heads:
                others = [t for hh, t in facts if hh != h]
                false_tail[h] = others[rng.integers(len(others))]

            b_feats = {layer: [] for layer in LAYERS}
            c_feats = {layer: [] for layer in LAYERS}
            overrides = 0
            for h in heads:
                noise = pick_noise_sentences(rel_name, spec, h, rng)
                doc_b = build_document(rel_name, spec, h, tails[h], noise)
                doc_c = build_document(rel_name, spec, h, false_tail[h], noise)
                prompt_b = f"Document: {doc_b} Q: {spec['q'].format(h=h)} A:"
                prompt_c = f"Document: {doc_c} Q: {spec['q'].format(h=h)} A:"

                fb = feats_for(tok, model, captured, prompt_b)
                fc = feats_for(tok, model, captured, prompt_c)
                for layer in LAYERS:
                    b_feats[layer].append(fb[layer]); c_feats[layer].append(fc[layer])

                completion = short_completion(tok, model, prompt_c)
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

    print("\n" + "=" * 88)
    print("  SUMMARY (mean over seeds)")
    print("=" * 88)
    print(f"\n  {'relation':<14}{'override %':>12}{'L12 gap (cross-same)':>24}{'L30 gap (cross-same)':>24}")
    for rel_name in RELATIONS_LARGE:
        ov = np.mean([o for _, _, o in summary[rel_name][LAYERS[0]]]) * 100
        gaps = {}
        for layer in LAYERS:
            vals = summary[rel_name][layer]
            gaps[layer] = np.mean([c - s for s, c, _ in vals])
        print(f"  {rel_name:<14}{ov:>11.0f}%{gaps[LAYERS[0]]:>24.4f}{gaps[LAYERS[1]]:>24.4f}")


if __name__ == "__main__":
    demo()
    run()
