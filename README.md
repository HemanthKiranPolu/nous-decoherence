# Late-Layer Geometric Resistance Under In-Context Knowledge Conflict

Workshop-note draft. Read [paper.md](paper.md) (or build `paper.pdf`, see below) for
the actual writeup — this README is just pointers.

**Honest framing, upfront:** this is a small/moderate-n exploratory interpretability
note, not a peer-reviewed, large-scale claim. The headline result does **not** survive
its own most important stress test (Section 2.6) — read that before citing anything
else here.

## What this is

A measured, replicated finding: on isolated single-sentence contradictions, when a
document contradicts a language model's parametric knowledge, a late-layer geometric
signal appears that is *not* explained by two initially-plausible mechanisms (per-fact
memorization strength; whether the model complies with the false claim) — both tested
directly and refuted (`paper.md` Sections 2.3, 2.4), and the direction replicates across
3 architecture families (Section 2.2). **But** scaled to realistic multi-sentence
documents, the effect survives for only 1 of 6 relations and reverses sign for the other
5 (Section 2.6) — the central limitation of the whole note, not a footnote.

## Build the PDF

```bash
pip install -r requirements.txt   # only needed to RUN the experiments, not to build the PDF
pandoc paper.md -o paper.pdf --pdf-engine=tectonic
```

## Reproduce the experiments

Each script in `nous/` is self-contained with a `demo()` self-check that runs before
the full experiment — see `paper.md` Section 5 for which script produces which result.
All experiments were run against local Hugging Face models (GPT-2, Qwen3-4B-Instruct,
Phi-3-mini-4k-instruct, Llama-3.2-3B-Instruct) on Apple Silicon (MPS). Llama-3 requires
an `HF_TOKEN` with an accepted license for `meta-llama/Llama-3.2-3B-Instruct`.

```bash
pip install -r requirements.txt
python -m nous.verifier_real_llm_test
```

## License

MIT (code). Paper text: feel free to cite, quote, or adapt with attribution.
