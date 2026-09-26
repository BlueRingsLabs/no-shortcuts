# Competency map

This is the contract. The course claims that if you complete it, you can work as a Data Engineer, Data
Scientist, Machine Learning Engineer, Deep Learning Engineer or AI Engineer. A claim like that needs to be
checkable, so here it is, skill by skill.

How I built this list: I took the skills that appear again and again in real job postings for these five roles
(the boring ones from banks and health insurers, not only the shiny ones from AI labs), the syllabi of the
university courses that have become the reference for each area (Stanford's CS229, CS231n, CS224n, CS336,
CS234; CMU's 10-414/714 Deep Learning Systems; Berkeley's Data 100), and the books that practitioners actually keep
on their desks (Kleppmann, Reis & Housley, Hastie/Tibshirani/Friedman, Bishop, Murphy, Goodfellow, Sutton & Barto,
Jurafsky & Martin, Huyen). Then I removed the buzzwords and kept the skills.

Each row says where the skill is taught and where you prove you have it. "Proven" means a lab with assertions
that fail if you got it wrong, or a capstone requirement. Reading a lesson proves nothing. You know that.

Lesson numbers look like `17.3` (module 17, lesson 3). Capstones are `A`, `B`, `C`, `D` (module 55).

---

## Shared core (all five roles)

You don't get to pick a role before you have these. Every senior person I respect in any of these five jobs has
all of them, even if some are rusty.

| Competency | Taught in | Proven by |
|---|---|---|
| Linux, shell, Git, virtual environments, Docker basics | 00.2, 07.4 | Every lab runs from the shell; capstones ship as containers |
| Production-grade Python: data model, typing, testing, packaging | 01.1, 01.2 | Lab 01.2 (a tested package), all capstones |
| Vectorized numerical code with NumPy | 01.3 | Labs 01.3, 02.x, 24.x |
| Profiling and concurrency | 01.4, 07.3 | Lab 01.4 |
| Linear algebra: vectors, matrices, least squares, eigen, SVD | 02.1 to 02.4 | Labs 02.x, 21.3 |
| Calculus and optimization: gradients, chain rule, GD, convexity, Lagrange/KKT | 03.1 to 03.4 | Labs 03.x, 19.3, 24.2 |
| Probability: distributions, Bayes, expectation, MLE/MAP | 04.1 to 04.4 | Labs 04.x |
| Statistics: estimation, bootstrap, hypothesis tests | 05.1, 05.2 | Labs 05.x |
| Information theory: entropy, cross-entropy, KL | 06.1, 06.2 | Lab 06.1 |
| Algorithms, complexity, numerical stability, hardware awareness | 07.1 to 07.3 | Labs 07.x |
| SQL, including window functions | 08.1, 08.2 | Labs 08.x |
| Data wrangling with pandas / Polars / DuckDB | 10.1, 10.2 | Labs 10.x |
| Security mindset: threat modeling, secrets, least privilege | 15.1, 44.1, 53.x | Capstone D (you attack your own system) |
| Writing: design docs, READMEs, post-mortems | 47.1, 56.1 | Every capstone requires a design doc and a report |

---

## Data Engineer

Builds and runs the systems that move, store, transform and serve data reliably, securely and cheaply.

| Competency | Taught in | Proven by |
|---|---|---|
| Relational model, SQL query tuning, query plans, indexes | 08.1 to 08.3 | Lab 08.3 |
| Transactions, isolation levels, OLTP vs OLAP | 08.4 | Lab 08.4 |
| Schema design: normalization, dimensional modeling, SCD type 1/2 | 09.1, 09.2 | Lab 09.2, Capstone A |
| Data cleaning, profiling and data quality tests | 10.3 | Lab 10.3, Capstone A |
| File formats: CSV, JSON, Avro, Parquet, Arrow; compression and encodings | 11.1 | Lab 11.1 |
| Object storage layouts, partitioning, small-files problem | 11.1, 11.2 | Lab 11.2 |
| Data lakes, warehouses, lakehouses, Iceberg/Delta, medallion architecture | 11.2 | Capstone A |
| Distributed processing: MapReduce model, Spark, shuffles, skew | 12.1, 12.2 | Labs 12.x |
| Choosing single-node (DuckDB/Polars) vs distributed | 10.2, 12.2 | Lab 12.2 |
| ETL/ELT design, idempotency, incremental loads, backfills | 13.1 | Lab 13.1, Capstone A |
| Orchestration (Airflow, Dagster): DAGs, retries, sensors, SLAs | 13.2 | Capstone A |
| dbt: models, tests, documentation | 13.3 | Capstone A |
| Streaming: Kafka, partitions, consumer groups, offsets | 14.1 | Lab 14.1 |
| Stream processing: event time, windows, watermarks, state, exactly-once, CDC | 14.2 | Lab 14.2 |
| Platform security: IAM, encryption, secrets, network boundaries | 15.1 | Capstone A security review |
| PII handling, privacy regulation, retention | 15.2, 54.2 | Capstone A |
| Catalogs, lineage, data contracts | 15.2 | Capstone A |
| Data observability, alerting, on-call, runbooks | 15.3 | Capstone A runbook |
| Feature pipelines and feature stores | 22.1, 49.1 | Capstone B |
| Cloud primitives, infrastructure as code, cost control | 52.1 | Capstone A cost report |

---

## Data Scientist

Turns data into decisions: analysis, experiments, statistical models and predictive models, communicated honestly.

| Competency | Taught in | Proven by |
|---|---|---|
| Exploratory data analysis and visualization that communicates | 10.4 | Lab 10.4 |
| Estimation, confidence intervals, bootstrap | 05.1 | Lab 05.1 |
| Hypothesis testing and multiple comparisons | 05.2 | Lab 05.2 |
| A/B test design, power analysis, sequential testing pitfalls | 05.3, 50.2 | Lab 05.3 |
| Causal inference: confounding, DAGs, matching, diff-in-diff, IV basics | 05.4 | Lab 05.4 |
| Problem framing, baselines, choosing the loss | 16.1, 47.1 | Capstone B design doc |
| Bias-variance, generalization, learning theory intuition | 16.2, 16.3 | Lab 16.2 |
| Linear and logistic regression, GLMs, regularization, derived and applied | 17.1 to 17.4 | Labs 17.x |
| Evaluation: CV schemes, metrics, calibration, thresholds, leakage | 18.1 to 18.4 | Labs 18.x |
| k-NN, Naive Bayes, discriminant analysis, SVMs | 19.1 to 19.3 | Labs 19.x |
| Trees, random forests, gradient boosting (XGBoost, LightGBM, CatBoost) | 20.1 to 20.4 | Labs 20.x, Capstone B |
| Clustering, GMM/EM, PCA, t-SNE/UMAP | 21.1 to 21.4 | Labs 21.x |
| Feature engineering and leak-free pipelines | 22.1 | Lab 22.1 |
| Hyperparameter optimization | 22.2 | Lab 22.2 |
| Interpretability: permutation importance, PDP, SHAP | 22.3 | Lab 22.3 |
| Time series forecasting and backtesting | 23.1 | Lab 23.1 |
| Recommender systems | 23.2 | Lab 23.2 |
| Anomaly detection | 23.3 | Lab 23.3 |
| Bayesian inference and Gaussian processes, uncertainty | 23.4 | Lab 23.4 |
| Fairness analysis | 54.1 | Lab 54.1, Capstone B |
| Communicating results to non-technical people | 10.4, 47.1, 56.1 | Capstone B report |

---

## Machine Learning Engineer

Gets models into production and keeps them there: pipelines, serving, testing, monitoring, cost, reliability.

| Competency | Taught in | Proven by |
|---|---|---|
| Everything in "Data Scientist" at a working level | Part III | Labs of Part III |
| Neural networks and PyTorch at a working level | 24.x, 25.x, 26.x | Labs 24.x to 26.x |
| ML system design and design docs | 47.1, 47.2 | Capstone B, D |
| Experiment tracking and model registry (MLflow) | 48.1 | Lab 48.1, Capstone B |
| Data and pipeline versioning, reproducible environments | 48.2 | Capstone B |
| Serving: batch, online, streaming; FastAPI model services | 49.1 | Lab 49.1, Capstone B |
| Containers and Kubernetes for ML, autoscaling | 49.2 | Capstone B |
| Inference optimization: ONNX, compilation, quantization, distillation, pruning | 49.3, 37.3 | Lab 49.3 |
| Testing ML code, data and models; CI/CD/CT | 50.1 | Capstone B CI pipeline |
| Deployment strategies: shadow, canary, A/B, bandits, rollback | 50.2, 46.1 | Capstone B |
| Monitoring: data drift, concept drift, delayed labels, retraining triggers | 51.1 | Lab 51.1, Capstone B |
| Feature stores and online/offline consistency | 22.1, 49.1 | Capstone B |
| Cloud, GPUs, IaC and cost | 52.1 | Capstone B cost report |
| Adversarial ML and ML supply chain security | 53.1, 53.2 | Labs 53.x |
| Responsible AI: fairness, privacy, documentation, regulation | 54.1 to 54.3 | Capstone B model card |

---

## Deep Learning Engineer

Designs, trains, debugs and scales neural networks, and understands them down to the kernel launch.

| Competency | Taught in | Proven by |
|---|---|---|
| Backpropagation derived and implemented; autograd engine from scratch | 24.1 to 24.3 | Labs 24.x |
| PyTorch internals: tensors, autograd, modules, data loading, training loops | 25.1, 25.2 | Labs 25.x |
| GPUs, mixed precision, torch.compile, reproducibility | 25.3, 33.3 | Lab 25.3 |
| Optimizers, initialization, schedules, training dynamics | 26.1, 26.2 | Labs 26.x |
| Regularization, normalization, residual connections | 26.3, 26.4 | Labs 26.x |
| Systematic debugging of training | 26.5 | Lab 26.5 |
| Convolutions and CNN architectures | 27.1, 27.2 | Labs 27.x |
| Transfer learning | 27.3 | Lab 27.3 |
| Object detection and segmentation | 27.4, 27.5 | Labs 27.4, 27.5 |
| RNNs, LSTMs, GRUs, seq2seq | 28.1, 28.2 | Labs 28.x |
| Attention and transformers from scratch; efficient attention | 29.1 to 29.3 | Labs 29.x |
| Self-supervised and contrastive learning; ViT; CLIP | 30.1, 30.2 | Labs 30.x |
| Generative models: VAE, GAN, diffusion, flow matching | 31.1 to 31.4 | Labs 31.x |
| Graph neural networks | 32.1 | Lab 32.1 |
| Memory and FLOP estimation, roofline thinking | 33.1, 07.3 | Lab 33.1 |
| Distributed training: DDP, FSDP/ZeRO, tensor and pipeline parallelism | 33.2 | Lab 33.2 |
| Profiling and speeding up training | 33.3 | Lab 33.3 |
| Language model pretraining end to end | 35.1 to 35.5 | Capstone C |

---

## AI Engineer

Builds products on top of foundation models: prompts, tools, retrieval, agents, evals, safety, cost.

| Competency | Taught in | Proven by |
|---|---|---|
| How LLMs work: tokenization, transformer decoders, pretraining, scaling | 35.1 to 35.5 | Labs 35.x |
| Encoders and embeddings models | 36.1, 41.1 | Labs 36.1, 41.1 |
| Decoding, KV cache, serving engines, quantization | 37.1 to 37.3 | Labs 37.x |
| Fine-tuning: SFT, LoRA, QLoRA | 38.1, 38.2 | Labs 38.x, Capstone C |
| Alignment: RLHF, DPO, reasoning models and RLVR | 38.3, 38.4 | Lab 38.3, Capstone C |
| Evaluating LLMs and detecting contamination | 39.1 | Lab 39.1 |
| Multimodal models | 39.2, 30.2 | Lab 30.2 |
| LLM APIs: streaming, retries, rate limits, token and cost accounting | 40.1 | Lab 40.1 |
| Prompt engineering and schema-validated structured outputs | 40.2 | Lab 40.2 |
| Tool use / function calling | 40.3 | Lab 40.3 |
| Semantic search and vector indexes (HNSW, IVF, PQ) | 41.1, 41.2 | Labs 41.1, 41.2 |
| RAG: chunking, hybrid search, reranking, citations | 41.3 | Lab 41.3, Capstone D |
| Advanced RAG and retrieval evaluation | 41.4 | Lab 41.4, Capstone D |
| Agents: ReAct loop, planning, memory, stopping | 42.1 | Lab 42.1 |
| MCP servers and clients, tool design | 42.2 | Lab 42.2, Capstone D |
| Multi-agent and workflow design | 42.3 | Capstone D |
| Eval suites and LLM-as-judge calibration | 43.1 | Lab 43.1, Capstone D |
| Tracing, monitoring and feedback loops for LLM apps | 43.2 | Capstone D |
| Prompt injection, OWASP LLM Top 10, red teaming, guardrails | 44.1, 44.2 | Labs 44.x, Capstone D |
| Gateways, caching, routing, cost control | 45.1 | Capstone D |
| Self-hosting open-weight models | 45.2 | Capstone D (optional track) |
| Reinforcement learning fundamentals (for post-training literacy) | 46.1 to 46.6 | Labs 46.x |

---

## What is deliberately out of scope

Being honest about the edges is part of the contract.

- **Research-level theory.** You'll get enough learning theory to reason about generalization (16.3), but not
  measure-theoretic probability or a full course in statistical learning theory. That's a PhD, and a PhD is a
  different product.
- **Writing CUDA kernels by hand.** You will understand what kernels do, why fusion matters and how to read a
  profiler (33.x), and you'll see Triton. Writing production CUDA is its own specialization.
- **Robotics and control.** RL here stops at the algorithms and at LLM post-training. Sim-to-real, control theory
  and hardware are not covered.
- **Speech synthesis and audio generation.** Speech recognition gets a section in 39.2. TTS doesn't.
- **Front-end development.** Your capstones will need a minimal UI. I'll point you at simple options and not
  teach React.
- **Vendor certifications.** No lesson here is "how to pass exam X". If you understand the material, the
  vendor exams are a weekend of reading their documentation. The reverse is not true.

If you think something essential is missing, open an issue with the job posting or the syllabus that proves it.
Evidence beats opinions. Even mine.
