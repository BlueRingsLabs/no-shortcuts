# Module 25: PyTorch

*Part IV: Deep Learning · about 25 hours*

The framework, properly. Not the tutorial, the framework.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What does `loss.backward()` do to `.grad`, and why do you call `zero_grad()`?

## When you finish it, you can

- Use tensors, autograd, views, broadcasting and devices without surprises.
- Write nn.Modules, Datasets, DataLoaders and a clean training loop with evaluation and checkpointing.
- Use GPUs, mixed precision and torch.compile, and make runs reproducible.

## Lessons

25.1. [Tensors and autograd](25.1-tensors-autograd.md)

25.2. [Modules, data loading and the training loop](25.2-modules-training-loop.md)

25.3. [GPUs, mixed precision and reproducibility](25.3-gpus-mixed-precision.md)

## Labs

Run them from the repository root, for example:

```bash
python part-4-deep-learning/25-pytorch/labs/lab_25_1_tensors_autograd.py
```

- [`lab_25_1_tensors_autograd.py`](labs/lab_25_1_tensors_autograd.py)
- [`lab_25_2_training_loop.py`](labs/lab_25_2_training_loop.py)
- [`lab_25_3_precision_reproducibility.py`](labs/lab_25_3_precision_reproducibility.py)

Back to [Part IV](../) · [Syllabus](../../SYLLABUS.md)
