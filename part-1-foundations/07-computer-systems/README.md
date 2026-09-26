# Module 07: Computer Systems for ML

*Part I: Foundations · about 35 hours*

Your model runs on a real machine with real memory, real floating point and a real network. Ignore that and it will remind you, usually in production.

## Self-check

If you can answer these without looking anything up, skim the lessons and go straight to the labs.
If you can't, good, that's what the module is for.

- Why does `np.exp(1000)` break softmax, and how do you fix it?
- What is the difference between latency and throughput?

## When you finish it, you can

- Pick the right data structure and estimate the time and memory complexity of your code.
- Explain IEEE 754 floating point, avoid catastrophic cancellation and overflow, and implement stable log-sum-exp and softmax.
- Reason about caches, memory bandwidth, CPU vs GPU execution and why some operations are memory-bound.
- Work with processes, signals, environment variables, HTTP and REST APIs on Linux.

## Lessons

07.1. [Algorithms and data structures ML engineers actually use](07.1-algorithms-data-structures.md)

07.2. [Floating point and numerical stability](07.2-floating-point.md)

07.3. [The machine: memory hierarchy, CPUs and GPUs](07.3-the-machine.md)

07.4. [Networks, APIs and Linux processes](07.4-networks-apis-linux.md)

## Labs

Run them from the repository root, for example:

```bash
python part-1-foundations/07-computer-systems/labs/lab_07_1_algorithms.py
```

- [`lab_07_1_algorithms.py`](labs/lab_07_1_algorithms.py)
- [`lab_07_2_floating_point.py`](labs/lab_07_2_floating_point.py)
- [`lab_07_3_the_machine.py`](labs/lab_07_3_the_machine.py)
- [`lab_07_4_networks_processes.py`](labs/lab_07_4_networks_processes.py)

Back to [Part I](../) · [Syllabus](../../SYLLABUS.md)
