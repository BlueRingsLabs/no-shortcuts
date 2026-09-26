# Module 14: Streaming

*Part II: Data Engineering · about 25 hours*

Data that doesn't wait for your nightly batch job.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- What is a watermark and what happens to events that arrive after it?

## When you finish it, you can

- Explain logs, topics, partitions, consumer groups and offsets in Kafka.
- Reason about event time vs processing time, windows, watermarks, state and delivery guarantees.
- Build a small streaming pipeline with change data capture.

## Lessons

14.1. [Logs, Kafka and event-driven architecture](14.1-kafka-event-driven.md)

14.2. [Stream processing: time, windows, state and exactly-once](14.2-stream-processing.md)

## Labs

Run them from the repository root, for example:

```bash
python part-2-data-engineering/14-streaming/labs/lab_14_1_partitioned_log.py
```

- [`lab_14_1_partitioned_log.py`](labs/lab_14_1_partitioned_log.py)
- [`lab_14_2_stream_processing.py`](labs/lab_14_2_stream_processing.py)

Back to [Part II](../) · [Syllabus](../../SYLLABUS.md)
