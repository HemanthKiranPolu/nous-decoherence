---
title: "Late-Layer Geometric Resistance Under In-Context Knowledge Conflict"
author: "Hemanth Kiran Polu"
date: "2026-07-06"
geometry: margin=1in
fontsize: 11pt
---

**Status:** workshop-note draft · CPU/MPS experiments · 2026-07-06
**Scope:** a narrow, falsification-heavy interpretability finding — see Section 5 before citing.

---

## Abstract

When a document in context contradicts a language model's parametric knowledge, does
that conflict leave a measurable trace in the residual stream? We test this on two
instruction-tuned models (Qwen3-4B-Instruct, Phi-3-mini) across three relation types
(country→capital, company→founder, book→author) and find a real, replicated signal:
at a late layer (~83% depth), the hidden state for a same-entity contradiction is
geometrically further from a clean baseline than switching to a different entity
entirely is; at a mid layer (~33% depth), there is no such gap. We then falsify our
own first two explanations for *why* the effect varies in size — first that it tracks
per-fact parametric memorization strength (rigidity hypothesis, refuted: Spearman
rho=0.006, n=45), then that it tracks the model successfully outputting the false claim
(active-lie hypothesis, refuted: resisting the contradiction costs *more* geometric
disruption than complying with it, n=73 vs 62). The mid-layer null is itself
reconciled with prior work: a *trained* probe recovers a real but modest signal
(AUROC 0.642) where our untrained geometric check found none, consistent with — but
quantitatively short of — Zhao et al. (2024)'s reported ~90% mid-layer probe accuracy.

## 1. Setup

Three relations, 15 facts each (45 total), each posed as `Document: {fact}. Q: {question}
A:` with either the true tail (congruent) or another fact's real tail substituted
(contradictory, a plausible hard negative rather than gibberish). Two models confirmed
to actually follow in-context overrides before use (checked directly, not assumed):
GPT-2-base and TinyLlama-1.1B-Chat both failed this premise outright and were excluded;
Qwen3-4B-Instruct and Phi-3-mini-4k-instruct both pass. Llama-3 itself is gated on
Hugging Face (manual approval required) and was not accessible in this environment.

## 2. Results

### 2.1 Real-LLM parametric recall (baseline sanity check)

A linear probe on GPT-2 hidden states, trained on 20 countries' true/false capital
sentences, reaches AUROC 0.980 on 10 held-out countries — confirms parametric facts are
linearly decodable and generalize to unseen entities, before testing anything harder.

### 2.2 Late-layer decoherence under in-context contradiction

Cosine similarity between congruent and contradictory hidden states, same entity, vs.
cosine similarity between two different entities' congruent hidden states (the "does a
lie perturb the representation more than an ordinary topic change" test), Qwen3-4B,
layers 12/36 and 30/36:

| layer | same-entity (congruent vs contradictory) | cross-entity (congruent vs congruent) |
|---|---|---|
| 12 | 0.9974 | 0.9969 |
| 30 | **0.9277** | **0.9829** |

Replicated across 3 relations × 3 seeds (mid-layer gap ~ 0 on all 9 runs, late-layer gap
positive on all 9), and on a second, architecturally different model (Phi-3-mini, layers
11/32 and 27/32; same qualitative pattern, 6/6 relation-model combinations agreeing on
direction). Relation-level magnitude is *not* stable across models: Qwen's largest effect
is `capital_of` (0.053), Phi-3's largest is `written_by` (0.043) — magnitude depends on
the model, not solely the relation.

### 2.3 Rigidity hypothesis — refuted at fact-level granularity

Proposed mechanism: facts held with a "deeper parametric attractor" should decohere more
under override. Tested directly: per-fact teacher-forced confidence in the true tail
(bare prompt, no document) against per-fact layer-30 decoherence, n=45 facts pooled
across all 3 relations. Spearman rho = 0.006 — no relationship. Confidence vs.
override-resistance: rho = −0.225, correct direction but below the ~0.29 threshold n=45
needs for significance. The relation-level magnitude pattern in Section 2.2 is real but not
explained by per-fact memorization strength.

### 2.4 Active-lie hypothesis — refuted with real statistical power

Proposed alternative: decoherence tracks the model successfully *committing* to the false
answer, not merely processing the contradiction. Tested on Phi-3 (lower compliance rate
gives a naturally large stubborn/compliant split: n=73 stubborn, n=62 compliant, vs. only
n=5 stubborn on Qwen):

| layer | stubborn (resisted override) | compliant (followed override) | baseline (no conflict) |
|---|---|---|---|
| 11 | 0.0036 | 0.0036 | 0.0095 |
| 27 | **0.1018** | **0.0747** | **0.0623** |

Stubborn > compliant > baseline at the late layer — the opposite of what an active-lie
mechanism predicts. Resisting an explicit contextual instruction to preserve a parametric
prior costs more geometric disruption than complying with it. At the mid layer, both
groups score *below* the cross-entity baseline, which is a lexical artifact (swapping the
whole subject perturbs shallow-layer tokens more than swapping only the claimed answer for
the same subject) rather than evidence about conflict processing.

### 2.5 Reconciling the mid-layer null with Zhao et al. (2024)

Zhao et al. (arXiv:2410.16090) report ~90% accuracy from a *trained* linear probe
detecting conflict-presence at intermediate layers of Llama3-8B — in apparent tension
with our untrained mid-layer null (Section 2.2). Replicating their actual method (trained
logistic regression, congruent=0/contradictory=1, held out on unseen entities) on our
45-fact Qwen set: AUROC 0.642 at layer 12, 1.000 at layer 30. A trained probe *does*
recover a real signal our untrained geometric check missed — the discrepancy is
substantially a method difference — but 0.642 falls well short of their ~90%, plausibly
due to this replication's much smaller dataset and a single train/held-out split with no
cross-validation, not a clean quantitative match.

## 3. What is and isn't shown

- A real, cross-model, cross-relation late-layer geometric signal under in-context
  contradiction exists and is not explained by per-fact memorization strength or by
  whether the model ultimately complies.
- The magnitude of the effect is model- and relation-dependent by a factor of 2–3×; no
  mechanism tested so far explains that variation.
- The practical framing that survives is **not** "lie detector" — it is closer to a
  **resistance/friction detector**: it appears to fire more when a model fights its own
  parametric prior than when it quietly updates to match new context.
- Not shown: what *does* explain the magnitude variation (relation type? entity type?
  something else); whether this transfers past 3 relation templates and 2 model families;
  whether it survives natural (non-templated) documents instead of single-sentence
  synthetic contradictions; a deployed detection threshold (none is proposed here).

## 4. Honest scope & limitations

- **Small n throughout.** 45 facts, 3 relations, 2–3 seeds per relation. Real effects, not
  yet large-sample claims.
- **Two working models, two failed premises.** GPT-2-base and TinyLlama-1.1B-Chat do not
  reliably follow in-context overrides at all and were excluded rather than forced; this
  itself is worth noting as a capability floor for any downstream use of this signal.
- **Llama-3 untested.** Gated, no license acceptance available in this environment.
- **Synthetic, single-sentence contradictions only.** Real RAG documents are longer, noisier,
  and multi-fact; this has not been tested on anything but templated single-fact conflicts.
- **No deployed detector.** This note establishes a measurable phenomenon, not a calibrated,
  cross-domain threshold ready to gate real generations.
- **Prior art.** Zhao et al. (2024) already show conflict-presence is probeable at
  intermediate layers; this note's distinct contribution is the late-layer
  resistance-vs-compliance split (Section 2.4) and the negative result on per-fact rigidity
  (Section 2.3), not the base claim that conflict leaves a residual-stream trace.

## 5. Reproduce

```bash
git clone https://github.com/HemanthKiranPolu/nous-decoherence
cd nous-decoherence
pip install -r requirements.txt
python -m nous.verifier_real_llm_test              # 2.1, GPT-2 parametric recall probe
python -m nous.verifier_in_context_conflict         # 2.2, Qwen single-relation decoherence
python -m nous.verifier_multi_relation_robustness   # 2.2, 3-relation x 3-seed Qwen sweep
python -m nous.rigidity_hypothesis_test             # 2.3, per-fact confidence vs decoherence
python -m nous.verifier_cross_model_phi3            # 2.2, Phi-3 cross-model replication
python -m nous.verifier_compliance_split_phi3       # 2.4, stubborn vs compliant split
python -m nous.replicate_zhao_conflict_probe        # 2.5, trained-probe reconciliation
```

## References

- Zhao et al., *Analysing the Residual Stream of Language Models Under Knowledge
  Conflicts*, arXiv:2410.16090 (2024).
- Li et al., *SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict
  Mitigation in Retrieval-Augmented Generation*, arXiv:2606.27786 (2026).
