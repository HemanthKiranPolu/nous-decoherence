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
that conflict leave a measurable trace in the residual stream? On isolated,
single-sentence synthetic contradictions, yes, robustly: across three instruction-tuned
models spanning three architecture families (Qwen3-4B-Instruct, Phi-3-mini,
Llama-3.2-3B-Instruct) and three relation types (country→capital, company→founder,
book→author), a late-layer (~82–83% depth) hidden state for a same-entity contradiction
is geometrically further from a clean baseline than switching to a different entity
entirely is (9/9 model×relation combinations agree on direction), while a mid-layer
(~32–34% depth) check shows no such gap. We then falsify our own first two explanations
for *why* the effect varies in size — first that it tracks per-fact parametric
memorization strength (rigidity hypothesis, refuted: Spearman rho=0.006, n=45), then that
it tracks the model successfully outputting the false claim (active-lie hypothesis,
refuted: resisting the contradiction costs *more* geometric disruption than complying
with it, n=73 vs 62). Finally — and this is the central limitation of the whole
result — the effect is largely an artifact of the single-sentence framing: scaled to 120
facts across 6 relations with realistic multi-sentence documents (the target fact
embedded among other true facts, not isolated), the late-layer gap survives for only 1 of
6 relations and *reverses sign* for the other 5. The mid-layer null is separately
reconciled with prior work: a *trained* probe recovers a real but modest signal
(AUROC 0.642) where our untrained geometric check found none, consistent with — but
quantitatively short of — Zhao et al. (2024)'s reported ~90% mid-layer probe accuracy.

## 1. Setup

Three relations, 15 facts each (45 total), each posed as `Document: {fact}. Q: {question}
A:` with either the true tail (congruent) or another fact's real tail substituted
(contradictory, a plausible hard negative rather than gibberish). Models confirmed to
actually follow in-context overrides before use (checked directly, not assumed, and at
the *aggregate* level, not on one example — a single Llama-3 test case looked like a
premise failure and was not representative, see Section 2.2): GPT-2-base and
TinyLlama-1.1B-Chat both failed this premise outright at scale and were excluded;
Qwen3-4B-Instruct, Phi-3-mini-4k-instruct, and Llama-3.2-3B-Instruct all pass, with
real (if uneven) override rates on every relation. Llama-3 required a license acceptance
and access token, obtained partway through this work.

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
positive on all 9), and on two further, architecturally different models: Phi-3-mini
(layers 11/32 and 27/32) and Llama-3.2-3B-Instruct (layers 9/28 and 23/28) — same
qualitative pattern on both, 9/9 relation-model combinations now agreeing on direction
across all three model families:

| model | relation | override % | mid-layer gap | late-layer gap |
|---|---|---|---|---|
| Qwen3-4B | capital_of | 71% | −0.0006 | **0.0525** |
| Qwen3-4B | founded_by | 98% | −0.0035 | **0.0156** |
| Qwen3-4B | written_by | 84% | −0.0048 | **0.0225** |
| Phi-3-mini | capital_of | 22% | −0.0021 | **0.0214** |
| Phi-3-mini | founded_by | 51% | −0.0067 | **0.0165** |
| Phi-3-mini | written_by | 64% | −0.0091 | **0.0430** |
| Llama-3.2-3B | capital_of | 89% | 0.0032 | **0.0612** |
| Llama-3.2-3B | founded_by | 16% | −0.0074 | **0.0816** |
| Llama-3.2-3B | written_by | 16% | −0.0156 | **0.1681** |

Relation-level magnitude is *not* stable across models: Qwen's largest effect is
`capital_of` (0.053), Phi-3's largest is `written_by` (0.043), Llama-3's largest is also
`written_by` but at 0.168 — nearly 3× Phi-3's — magnitude depends on the model, not
solely the relation, and Llama-3's effect sizes are uniformly the largest of the three
models on this single-sentence framing.

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

### 2.6 Realistic multi-sentence documents — the effect mostly does not survive

Every result above uses an isolated, single-sentence document: `Document: France is a
country whose capital is Munich.` Real RAG documents are longer and contain other true
content alongside the fact being queried. Scaled to 120 facts across 6 relations
(`capital_of`, `founded_by`, `written_by`, `painted_by`, `directed_by`, `composed_by`,
20 facts each) with documents built as an opener + 2 *other true facts from the same
relation* + the target sentence + a closer (congruent and contradictory versions share
the identical filler facts, differing only in the target sentence's tail — verified by
construction, and a first version of this experiment had a bug letting filler facts
differ between the two versions; fixed and rerun before trusting the result below),
Qwen3-4B-Instruct:

| relation | override % | mid-layer gap | late-layer gap |
|---|---|---|---|
| capital_of | 35% | −0.0011 | **0.0048** |
| founded_by | 58% | −0.0032 | **−0.0105** |
| written_by | 88% | −0.0040 | **−0.0177** |
| painted_by | 92% | −0.0039 | **−0.0179** |
| directed_by | 85% | −0.0033 | **−0.0200** |
| composed_by | 95% | −0.0050 | **−0.0171** |

The late-layer gap survives for only 1 of 6 relations (`capital_of`, and even there it
shrinks roughly 10× from the single-sentence version's 0.0525 to 0.0048) and *reverses
sign* for the other 5. This is the central limitation of the whole result: the effect
documented in Sections 2.2–2.4 is largely specific to isolated single-sentence
contradictions and does not straightforwardly generalize to documents containing other
true content alongside the target fact — the realistic case a deployed RAG verifier
would actually need to handle. A plausible mechanism, not yet tested directly: once a
document contains multiple true facts alongside one manipulated one, most of the
document's content is identical between the congruent and contradictory versions, which
may dominate the late-layer representation and wash out or invert the effect of the one
changed sentence relative to switching to a document about a completely different entity.

## 3. What is and isn't shown

- On isolated single-sentence contradictions, a real, cross-model (3 architecture
  families), cross-relation late-layer geometric signal exists and is not explained by
  per-fact memorization strength or by whether the model ultimately complies.
- **That signal mostly does not survive realistic multi-sentence documents** (Section 2.6)
  — 1 of 6 relations keeps a (much smaller) positive gap, 5 reverse sign. This is the
  headline limitation, not a footnote: whatever this signal is, it is substantially a
  property of isolated single-fact framing, not yet a property of realistic documents.
- The magnitude of the single-sentence effect is model- and relation-dependent by a
  factor of up to ~3×; no mechanism tested so far explains that variation, and it is now
  moot for 5 of 6 relations once documents are realistic anyway.
- The practical framing that survives on single-sentence documents is **not** "lie
  detector" — it is closer to a **resistance/friction detector**: it appears to fire more
  when a model fights its own parametric prior than when it quietly updates to match new
  context. Whether *any* framing survives on realistic documents is now the open question.
- Not shown: what explains the magnitude variation on single-sentence documents; what
  explains the sign reversal on realistic documents; whether either transfers past 6
  relation templates and 3 model families; a deployed detection threshold (none is
  proposed here, and Section 2.6 makes one look premature).

## 4. Honest scope & limitations

- **The realistic-document result (Section 2.6) is the main limitation, not an
  afterthought.** The signal this note is built around is largely specific to isolated
  single-sentence contradictions. Anyone citing the Section 2.2–2.4 numbers should cite
  Section 2.6 in the same breath.
- **Moderate n.** 120 facts / 6 relations for the realistic-document result; 45 facts / 3
  relations for the single-sentence cross-model result. Real effects at this scale, not
  yet large-sample claims.
- **Two models failed the override premise entirely.** GPT-2-base and TinyLlama-1.1B-Chat
  do not reliably follow in-context overrides at all and were excluded rather than forced;
  worth noting as a capability floor for any downstream use of this signal. A single
  worked example is not sufficient evidence a model passes this premise either way — one
  Llama-3 example looked like a failure and was not representative of its 45-fact
  aggregate rate (Section 2.2).
- **No deployed detector.** This note establishes a measurable phenomenon (fragile even
  on the axis it was found on), not a calibrated, cross-domain threshold ready to gate
  real generations.
- **Prior art.** Zhao et al. (2024) already show conflict-presence is probeable at
  intermediate layers; this note's distinct contributions are the late-layer
  resistance-vs-compliance split (Section 2.4), the negative result on per-fact rigidity
  (Section 2.3), and the negative result on realistic-document generalization
  (Section 2.6) — not the base claim that conflict leaves a residual-stream trace.

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
python -m nous.verifier_cross_model_llama3          # 2.2, Llama-3 cross-model replication (HF_TOKEN required)
python -m nous.verifier_compliance_split_phi3       # 2.4, stubborn vs compliant split
python -m nous.replicate_zhao_conflict_probe        # 2.5, trained-probe reconciliation
python -m nous.verifier_scaled_realistic            # 2.6, 120-fact realistic-document sweep
```

## 6. What would make this result matter again

Section 2.6 is not a call for more relations or more seeds at the same design — it is a
falsification of the practical version of the claim. Restoring it requires one of two
things, not incremental replication of what is already here:

1. **A signal that survives realistic documents.** Not this cosine-similarity-on-raw-
   activations measurement at a fixed layer, evaluated on documents with other true
   content around the target fact — something that isolates the target fact's
   contribution from the filler's, rather than comparing whole-document hidden states
   where filler content dominates by volume.
2. **A mechanistic account of the collapse that generalizes.** "The identical filler
   facts probably dominate the representation" (Section 2.6) is a plausible guess, stated
   as one, not tested. A real account would predict *in advance* — from document length,
   filler/target ratio, or something else measurable — which relations keep a signal and
   which invert, rather than describing the six numbers already observed after the fact.

Neither is a small follow-up experiment. Absent one of these, the honest summary of this
note is: a real, replicated, mechanistically-unexplained artifact of single-sentence
framing, refuted twice on its proposed mechanism, and refuted once on its practical
applicability. That is a legitimate negative result. It is not, on the evidence collected
here, a verifier.

## References

- Zhao et al., *Analysing the Residual Stream of Language Models Under Knowledge
  Conflicts*, arXiv:2410.16090 (2024).
- Li et al., *SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict
  Mitigation in Retrieval-Augmented Generation*, arXiv:2606.27786 (2026).
