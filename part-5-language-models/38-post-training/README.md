# Module 38: Post-Training

*Part V: Language Models · about 40 hours*

How a text predictor becomes an assistant, and how assistants learned to reason.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why does RLHF include a KL penalty to the reference model?

## When you finish it, you can

- Run supervised fine-tuning with chat templates and loss masking.
- Implement LoRA and use QLoRA to fine-tune on consumer hardware.
- Explain RLHF (reward models, PPO with KL penalty) and derive and implement DPO.
- Explain reasoning models: chain of thought, RL with verifiable rewards, GRPO and test-time compute.

## Lessons

38.1. [Supervised fine-tuning and instruction tuning](38.1-sft.md)

38.2. [Parameter-efficient fine-tuning: LoRA and QLoRA](38.2-lora-qlora.md)

38.3. [Preference optimization: RLHF and DPO](38.3-rlhf-dpo.md)

38.4. [Reasoning models and RL with verifiable rewards](38.4-reasoning-rlvr.md)

## Labs

Run them from the repository root, for example:

```bash
python part-5-language-models/38-post-training/labs/lab_38_1_sft.py
```

- [`lab_38_1_sft.py`](labs/lab_38_1_sft.py)
- [`lab_38_2_lora.py`](labs/lab_38_2_lora.py)
- [`lab_38_3_preferences.py`](labs/lab_38_3_preferences.py)
- [`lab_38_4_reasoning_rlvr.py`](labs/lab_38_4_reasoning_rlvr.py)

Back to [Part V](../) · [Syllabus](../../SYLLABUS.md)
