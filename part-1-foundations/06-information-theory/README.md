# Module 06: Information Theory

*Part I: Foundations · about 15 hours*

Short module, huge payoff. Cross-entropy loss, KL divergence, perplexity and half of the generative modeling literature suddenly make sense.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why is KL divergence not symmetric, and when does that matter?

## When you finish it, you can

- Compute entropy, cross-entropy and KL divergence and explain them as coding costs.
- Explain why classifiers and language models are trained with cross-entropy, and what perplexity measures.
- Use mutual information and understand the compression view of learning.

## Lessons

06.1. [Entropy, cross-entropy and KL divergence](06.1-entropy-kl.md)

06.2. [Mutual information and learning as compression](06.2-mutual-information-compression.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/06-information-theory/labs/lab_06_1_entropy.py
```

- [`lab_06_1_entropy.py`](labs/lab_06_1_entropy.py)
- [`lab_06_2_mutual_information.py`](labs/lab_06_2_mutual_information.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)
