# Late-Layer Geometric Resistance Under In-Context Knowledge Conflict

Workshop-note draft. Read [paper.md](paper.md) (or build `paper.pdf`, see below) for
the actual writeup — this README is just pointers.

**Honest framing, upfront:** this is a small-n (45 facts, 2 model families) exploratory
interpretability note, not a peer-reviewed, large-scale claim. Section 4 of the paper
states the limitations plainly. Read those before citing anything here.

## What this is

A measured, replicated finding: when a document in context contradicts a language
model's parametric knowledge, a late-layer geometric signal appears that is *not*
explained by two initially-plausible mechanisms (per-fact memorization strength;
whether the model complies with the false claim) — both were tested directly and
refuted. See `paper.md` Sections 2.3 and 2.4.

## Build the PDF

```bash
pip install -r requirements.txt   # only needed to RUN the experiments, not to build the PDF
pandoc paper.md -o paper.pdf --pdf-engine=tectonic
```

## Reproduce the experiments

Each script in `nous/` is self-contained with a `demo()` self-check that runs before
the full experiment — see `paper.md` Section 5 for which script produces which result.
All experiments were run against local Hugging Face models (GPT-2, Qwen3-4B-Instruct,
Phi-3-mini-4k-instruct) on Apple Silicon (MPS). Llama-3 was not testable: gated on
Hugging Face, no license acceptance available at run time.

```bash
pip install -r requirements.txt
python -m nous.verifier_real_llm_test
```

## License

MIT (code). Paper text: feel free to cite, quote, or adapt with attribution.
