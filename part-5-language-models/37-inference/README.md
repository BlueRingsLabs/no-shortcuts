# Module 37: Inference

*Part V: Language Models · about 25 hours*

Training happens once. Inference happens every time a user types something. That's where the money goes.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- How big is the KV cache for a 32-layer model with 8 KV heads of dim 128 at 8k context in fp16?

## When you finish it, you can

- Implement greedy, beam, temperature, top-k and top-p decoding, and constrained decoding.
- Explain the KV cache, continuous batching, PagedAttention and speculative decoding.
- Quantize models (int8, int4, GPTQ, AWQ, GGUF) and measure the quality cost.

## Lessons

37.1. [Decoding strategies](37.1-decoding.md)

37.2. [KV cache, batching and serving engines](37.2-kv-cache-serving.md)

37.3. [Quantization](37.3-quantization.md)

## Labs

Run them from the repository root, for example:

```bash
python part-5-language-models/37-inference/labs/lab_37_1_decoding.py
```

- [`lab_37_1_decoding.py`](labs/lab_37_1_decoding.py)
- [`lab_37_2_kv_cache_serving.py`](labs/lab_37_2_kv_cache_serving.py)
- [`lab_37_3_quantization.py`](labs/lab_37_3_quantization.py)

Back to [Part V](../) · [Syllabus](../../SYLLABUS.md)
