# Module 26: Training Deep Networks

*Part IV: Deep Learning · about 40 hours*

Why the same architecture trains beautifully for one person and diverges for another.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is the difference between L2 regularization and decoupled weight decay in Adam?

## When you finish it, you can

- Implement and compare SGD, momentum, Nesterov, RMSProp, Adam and AdamW.
- Choose initializations and learning-rate schedules (warmup, cosine, one-cycle) and read training curves.
- Regularize with weight decay, dropout, data augmentation, label smoothing and early stopping.
- Explain BatchNorm, LayerNorm, RMSNorm and residual connections, and why they make deep nets trainable.
- Debug training systematically: overfit one batch, check gradients, check data, check the loss.

## Lessons

26.1. [Optimizers: SGD, momentum, Adam, AdamW](26.1-optimizers.md)

26.2. [Initialization, learning rate schedules and training dynamics](26.2-init-schedules-dynamics.md)

26.3. [Regularization: weight decay, dropout, augmentation](26.3-regularization-dl.md)

26.4. [Normalization and residual connections](26.4-normalization-residuals.md)

26.5. [A recipe for training and debugging neural networks](26.5-training-recipe.md)

## Labs

Run them from the repository root, for example:

```bash
python part-4-deep-learning/26-training-deep-networks/labs/lab_26_1_optimizers.py
```

- [`lab_26_1_optimizers.py`](labs/lab_26_1_optimizers.py)
- [`lab_26_2_init_schedules.py`](labs/lab_26_2_init_schedules.py)
- [`lab_26_3_regularization.py`](labs/lab_26_3_regularization.py)
- [`lab_26_4_norm_residual.py`](labs/lab_26_4_norm_residual.py)
- [`lab_26_5_debugging.py`](labs/lab_26_5_debugging.py)

Back to [Part IV](../) · [Syllabus](../../SYLLABUS.md)
