"""Tests the specific claim proposed after verifier_multi_relation_robustness.py:
that late-layer decoherence magnitude is proportional to how rigidly/parametrically
a fact is memorized ("deep, sharp attractor" for capital_of vs "wide, shallow" for
founded_by/written_by).

The proposed way to test this needed pretraining-frequency or "parametric loss"
data we don't have (Qwen's training corpus isn't public). The proxy we DO have:
the model's own teacher-forced confidence in the true tail, given a bare
parametric prompt with no document -- a standard proxy for "how strongly is this
fact memorized" in the fact-editing literature (ROME/MEMIT use exactly this kind
of measure). That's computable directly, at PER-FACT granularity (n=45 across 3
relations) rather than the previous test's per-relation aggregate (n=3 relations,
too few to call anything a correlation).

Per fact (h, t): compute
  - parametric confidence: mean per-token log-prob of the true tail, given
    "Q: {question}? A:" with no document (higher = model already "knows" the
    fact strongly on its own, before any context is involved).
  - decoherence: 1 - cosine_similarity(B, C) at layer 30 (B=congruent-document
    hidden state, C=contradictory-document hidden state), same construction as
    verifier_in_context_conflict.py, single seed, per fact (not averaged away).
  - override: did the model actually follow the contradictory document (binary).

Then Spearman-correlate confidence against decoherence and against override,
pooled across all 45 facts -- this is the actual rigidity hypothesis, tested at
the granularity that can support a real correlation claim.

Run: python -m nous.rigidity_hypothesis_test
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nous.verifier_multi_relation_robustness import RELATIONS, LAYERS

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LAYER = 30
SEED = 0


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("mps")
    model.eval()
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out)[0, -1, :].float().clone()

    model.model.layers[LAYER].register_forward_hook(hook)
    return tok, model, captured


@torch.no_grad()
def feats_for(tok, model, captured, prompt: str) -> np.ndarray:
    ids = tok(prompt, return_tensors="pt").to("mps")
    model(**ids)
    return captured["h"].cpu().numpy()


@torch.no_grad()
def parametric_confidence(tok, model, prompt: str, tail: str) -> float:
    """Mean per-token log-prob of `tail`'s tokens, teacher-forced after `prompt`
    (no document, bare recall) -- higher means the model needs less help to
    produce this fact on its own."""
    full = prompt + " " + tail
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to("mps")
    full_ids = tok(full, return_tensors="pt").input_ids.to("mps")
    tail_ids = full_ids[0, prompt_ids.shape[1]:]
    if tail_ids.numel() == 0:
        return float("nan")
    logits = model(full_ids).logits[0]
    log_probs = F.log_softmax(logits[prompt_ids.shape[1] - 1: -1], dim=-1)
    token_lp = log_probs[torch.arange(tail_ids.numel()), tail_ids]
    return token_lp.mean().item()


@torch.no_grad()
def short_completion(tok, model, prompt: str, max_new_tokens: int = 14) -> str:
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def demo() -> None:
    """Self-check: confidence is higher for a well-known fact than a scrambled
    one, on one example, before trusting the full 45-fact sweep."""
    tok, model, captured = load_model()
    q = "Q: What is the capital of France? A:"
    c_true = parametric_confidence(tok, model, q, "Paris")
    c_false = parametric_confidence(tok, model, q, "Zzyzxville")
    assert c_true > c_false


def run() -> None:
    print("=" * 84)
    print(f"  Rigidity hypothesis: parametric confidence vs layer-{LAYER} decoherence, per fact")
    print("=" * 84)

    tok, model, captured = load_model()
    rng = np.random.default_rng(SEED)

    rows = []  # (relation, head, confidence, decoherence, override)
    for rel_name, spec in RELATIONS.items():
        facts = spec["facts"]
        heads = [h for h, _ in facts]
        tails = {h: t for h, t in facts}
        false_tail = {}
        for h in heads:
            others = [t for hh, t in facts if hh != h]
            false_tail[h] = others[rng.integers(len(others))]

        for h in heads:
            bare_q = f"Q: {spec['q'].format(h=h)} A:"
            conf = parametric_confidence(tok, model, bare_q, tails[h])

            doc_true = f"Document: {spec['doc'].format(h=h, t=tails[h])} Q: {spec['q'].format(h=h)} A:"
            doc_false = f"Document: {spec['doc'].format(h=h, t=false_tail[h])} Q: {spec['q'].format(h=h)} A:"
            fb = feats_for(tok, model, captured, doc_true)
            fc = feats_for(tok, model, captured, doc_false)
            sim = float(np.dot(fb, fc) / (np.linalg.norm(fb) * np.linalg.norm(fc)))
            decoherence = 1 - sim

            completion = short_completion(tok, model, doc_false)
            overridden = int(false_tail[h] in completion and tails[h] not in completion)

            rows.append((rel_name, h, conf, decoherence, overridden))

    print(f"\n  {'relation':<12}{'head':<24}{'confidence':>12}{'decoherence':>14}{'overridden':>12}")
    for rel_name, h, conf, dec, ov in rows:
        print(f"  {rel_name:<12}{h:<24}{conf:>12.3f}{dec:>14.5f}{ov:>12d}")

    conf_arr = np.array([r[2] for r in rows])
    dec_arr = np.array([r[3] for r in rows])
    ov_arr = np.array([r[4] for r in rows])

    print("\n" + "=" * 84)
    print(f"  n = {len(rows)} facts, pooled across {len(RELATIONS)} relations")
    print(f"  Spearman(confidence, decoherence) = {spearman(conf_arr, dec_arr):.3f}"
          f"   (rigidity hypothesis predicts POSITIVE: more confident -> more decoherence)")
    print(f"  Spearman(confidence, override)    = {spearman(conf_arr, ov_arr):.3f}"
          f"   (predicts NEGATIVE: more confident -> resists override more)")
    print("=" * 84)


if __name__ == "__main__":
    demo()
    run()
