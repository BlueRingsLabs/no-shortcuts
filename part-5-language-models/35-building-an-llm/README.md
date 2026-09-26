# Module 35: Building an LLM

*Part V: Language Models · about 50 hours*

The whole stack of a decoder-only language model, minus the hundred million dollars.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Under Chinchilla-style scaling, roughly how many tokens should a 1B parameter model see?

## When you finish it, you can

- Implement byte-level BPE tokenization.
- Implement and train a GPT-style decoder-only transformer.
- Explain pretraining data pipelines: crawling, extraction, filtering, deduplication, mixing, contamination.
- Use scaling laws to budget compute, parameters and tokens.
- Explain the modern LLM architecture: RoPE, RMSNorm, SwiGLU, GQA, mixture of experts.

## Lessons

35.1. [Tokenization: BPE from scratch](35.1-bpe-tokenizer.md)

35.2. [A GPT from scratch](35.2-gpt-from-scratch.md)

35.3. [Pretraining data](35.3-pretraining-data.md)

35.4. [Scaling laws and compute budgets](35.4-scaling-laws.md)

35.5. [The modern LLM architecture: RoPE, RMSNorm, SwiGLU, GQA, MoE](35.5-modern-architecture.md)

## Labs

Run them from the repository root, for example:

```bash
python part-5-language-models/35-building-an-llm/labs/lab_35_1_bpe.py
```

- [`lab_35_1_bpe.py`](labs/lab_35_1_bpe.py)
- [`lab_35_2_gpt.py`](labs/lab_35_2_gpt.py)
- [`lab_35_3_pretraining_data.py`](labs/lab_35_3_pretraining_data.py)
- [`lab_35_4_scaling_laws.py`](labs/lab_35_4_scaling_laws.py)
- [`lab_35_5_modern_architecture.py`](labs/lab_35_5_modern_architecture.py)

Back to [Part V](../) · [Syllabus](../../SYLLABUS.md)
