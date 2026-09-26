# Module 49: Serving

*Part VIII: MLOps and ML Systems · about 30 hours*

Getting predictions to the people who need them, fast and cheaply.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Where does p99 latency come from in a model server?

## When you finish it, you can

- Choose between batch, online and streaming inference and implement an online service with FastAPI.
- Containerize models with Docker and deploy them on Kubernetes with autoscaling.
- Optimize models for inference with ONNX, TensorRT-style compilation, distillation and pruning.

## Lessons

49.1. [Serving patterns: batch, online, streaming](49.1-serving-patterns.md)

49.2. [Containers and Kubernetes for ML](49.2-containers-kubernetes.md)

49.3. [Making models small and fast: ONNX, TensorRT, distillation, pruning](49.3-small-fast-models.md)

## Labs

Run them from the repository root, for example:

```bash
python part-8-mlops/49-serving/labs/lab_49_1_serving_patterns.py
```

- [`lab_49_1_serving_patterns.py`](labs/lab_49_1_serving_patterns.py)
- [`lab_49_2_containers_k8s.py`](labs/lab_49_2_containers_k8s.py)
- [`lab_49_3_small_fast_models.py`](labs/lab_49_3_small_fast_models.py)

Back to [Part VIII](../) · [Syllabus](../../SYLLABUS.md)
