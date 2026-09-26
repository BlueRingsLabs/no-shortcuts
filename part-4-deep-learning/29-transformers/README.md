# Module 29: Transformers

*Part IV: Deep Learning · about 35 hours*

The architecture behind almost everything interesting since 2017, built from the ground up.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why do we divide by sqrt(d_k) in attention?

## When you finish it, you can

- Derive scaled dot-product attention and multi-head attention and implement them.
- Build a transformer block (pre-norm, residuals, MLP) from scratch and train it.
- Explain positional encodings (sinusoidal, learned, RoPE, ALiBi) and efficient attention (FlashAttention, sparse, linear).

## Lessons

29.1. [Attention from first principles](29.1-attention.md)

29.2. [The transformer block, built from scratch](29.2-transformer-block.md)

29.3. [Positions, efficiency and FlashAttention](29.3-positions-efficiency.md)

## Labs

Run them from the repository root, for example:

```bash
python part-4-deep-learning/29-transformers/labs/lab_29_1_attention.py
```

- [`lab_29_1_attention.py`](labs/lab_29_1_attention.py)
- [`lab_29_2_transformer_block.py`](labs/lab_29_2_transformer_block.py)
- [`lab_29_3_positions_efficiency.py`](labs/lab_29_3_positions_efficiency.py)

Back to [Part IV](../) · [Syllabus](../../SYLLABUS.md)
