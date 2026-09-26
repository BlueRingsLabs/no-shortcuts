# Module 53: Security of ML Systems

*Part VIII: MLOps and ML Systems · about 20 hours*

The attack surface nobody put in the architecture diagram.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why is `torch.load` on an untrusted file a remote code execution vulnerability?

## When you finish it, you can

- Implement and defend against evasion attacks (FGSM, PGD), poisoning and backdoors, model extraction and membership inference.
- Secure the ML supply chain: unsafe serialization (pickle), model hubs, dependencies, SBOMs and signing.

## Lessons

53.1. [Adversarial machine learning](53.1-adversarial-ml.md)

53.2. [The ML supply chain](53.2-ml-supply-chain.md)

## Labs

Run them from the repository root, for example:

```bash
python part-8-mlops/53-ml-security/labs/lab_53_1_adversarial_ml.py
```

- [`lab_53_1_adversarial_ml.py`](labs/lab_53_1_adversarial_ml.py)
- [`lab_53_2_supply_chain.py`](labs/lab_53_2_supply_chain.py)

Back to [Part VIII](../) · [Syllabus](../../SYLLABUS.md)
