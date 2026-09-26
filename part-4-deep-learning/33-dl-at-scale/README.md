# Module 33: Deep Learning at Scale

*Part IV: Deep Learning · about 30 hours*

What changes when the model doesn't fit on one GPU and the bill has six digits.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- How much GPU memory does full fine-tuning of a 7B model with Adam in mixed precision need, roughly?

## When you finish it, you can

- Estimate FLOPs, memory (weights, gradients, optimizer state, activations) and arithmetic intensity for a training run.
- Explain and use data parallelism (DDP), sharded training (FSDP/ZeRO), tensor and pipeline parallelism.
- Profile training, use activation checkpointing, fused kernels and torch.compile.

## Lessons

33.1. [GPU arithmetic: memory, FLOPs and bandwidth](33.1-gpu-arithmetic.md)

33.2. [Distributed training: DDP, FSDP, tensor and pipeline parallelism](33.2-distributed-training.md)

33.3. [Profiling and making it fast](33.3-profiling-speed.md)

## Labs

Run them from the repository root, for example:

```bash
python part-4-deep-learning/33-dl-at-scale/labs/lab_33_1_gpu_arithmetic.py
```

- [`lab_33_1_gpu_arithmetic.py`](labs/lab_33_1_gpu_arithmetic.py)
- [`lab_33_2_distributed.py`](labs/lab_33_2_distributed.py)
- [`lab_33_3_profiling.py`](labs/lab_33_3_profiling.py)

Back to [Part IV](../) · [Syllabus](../../SYLLABUS.md)
