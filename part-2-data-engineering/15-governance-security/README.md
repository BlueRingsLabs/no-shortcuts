# Module 15: Governance, Security and Reliability

*Part II: Data Engineering · about 25 hours*

Your data platform is the most valuable target in the company. Treat it that way.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Where do credentials usually leak in a data platform?

## When you finish it, you can

- Threat-model a data platform: IAM, least privilege, encryption, secrets, network boundaries.
- Handle PII correctly: classification, masking, retention, and the basics of GDPR-style obligations.
- Use catalogs, lineage and data contracts to keep producers and consumers honest.
- Run data observability: freshness, volume, schema and distribution checks, alerting and on-call.

## Lessons

15.1. [Securing data platforms](15.1-securing-data-platforms.md)

15.2. [Privacy, lineage, catalogs and data contracts](15.2-privacy-lineage-contracts.md)

15.3. [Data observability and on-call](15.3-observability-oncall.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/15-governance-security/labs/lab_15_1_platform_security.py
```

- [`lab_15_1_platform_security.py`](labs/lab_15_1_platform_security.py)
- [`lab_15_2_privacy_lineage.py`](labs/lab_15_2_privacy_lineage.py)
- [`lab_15_3_data_observability.py`](labs/lab_15_3_data_observability.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
