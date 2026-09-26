# Syllabus

*No Shortcuts: Data, ML, DL and AI Engineering*, by Gonzalo L. Romero (DeepRat)

This file is generated from `course.yaml` by `tools/build_docs.py`. Don't edit it by hand, I will
overwrite your changes the next time I add a lesson and you will be sad.

**10 parts, 57 modules, 173 lessons, about 1,825 hours of
guided study.** Add roughly 30% on top for the time you will spend stuck on something stupid. That time
is not wasted, it's the part where you actually learn.

The hours are my honest estimate for someone who does every exercise and every lab. If you only read,
divide by three, and then also divide what you retain by three.

## Map

| Part | Modules | Hours |
|---|---|---|
| [0. Orientation](part-0-orientation/) | 00 | 10 |
| [I. Foundations](part-1-foundations/) | 01, 02, 03, 04, 05, 06, 07 | 255 |
| [II. Data Engineering](part-2-data-engineering/) | 08, 09, 10, 11, 12, 13, 14, 15 | 230 |
| [III. Classical Machine Learning](part-3-classical-ml/) | 16, 17, 18, 19, 20, 21, 22, 23 | 245 |
| [IV. Deep Learning](part-4-deep-learning/) | 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 | 295 |
| [V. Language Models](part-5-language-models/) | 34, 35, 36, 37, 38, 39 | 175 |
| [VI. AI Engineering](part-6-ai-engineering/) | 40, 41, 42, 43, 44, 45 | 150 |
| [VII. Reinforcement Learning](part-7-reinforcement-learning/) | 46 | 50 |
| [VIII. MLOps and ML Systems](part-8-mlops/) | 47, 48, 49, 50, 51, 52, 53, 54 | 165 |
| [IX. Capstones and Career](part-9-capstones/) | 55, 56 | 250 |

## Part 0: Orientation

Three lessons that save you three months. What the job titles really mean, how to set up a machine you won't have to fight, and how to study something this big without lying to yourself.

### Module 00: [Before You Start](part-0-orientation/00-before-you-start/) (10 h)

The boring part that everybody skips and then regrets. Read it anyway.

**When you finish it, you can:**

- Explain what a Data Engineer, Data Scientist, ML Engineer, DL Engineer and AI Engineer do all day, and where the lines blur.
- Run a Linux or WSL workstation with Git, uv, Python 3.11+, Docker and an editor that doesn't get in your way.
- Plan your study with retrieval practice and projects instead of passive video watching.

**Lessons:**

- 00.1 [What the job titles actually mean](part-0-orientation/00-before-you-start/00.1-job-titles.md)
- 00.2 [Your workstation: Linux, shell, Git, Python environments, Docker](part-0-orientation/00-before-you-start/00.2-workstation.md)
- 00.3 [How to study this without lying to yourself](part-0-orientation/00-before-you-start/00.3-how-to-study.md)

## Part I: Foundations

Python, linear algebra, calculus, probability, statistics, information theory and computer systems. This is the part people want to skip. It is also the part that separates engineers who understand why their model broke from engineers who restart the kernel and pray.

### Module 01: [Python for Engineers](part-1-foundations/01-python-for-engineers/) (40 h)

Not "Python for beginners". Python for people who will write code that other people run at 3 AM.

**When you finish it, you can:**

- Use Python's data model (dunder methods, iterators, generators, context managers) to write idiomatic code.
- Write typed, tested, packaged code with pytest and a pyproject.toml.
- Think in arrays: replace Python loops with NumPy vectorized operations and understand broadcasting and memory layout.
- Profile code, find the real bottleneck, and choose between threads, processes and asyncio.

**Lessons:**

- 01.1 [Python's data model and the idioms that matter](part-1-foundations/01-python-for-engineers/01.1-data-model-and-idioms.md)
- 01.2 [Types, tests and packages: code other people can trust](part-1-foundations/01-python-for-engineers/01.2-types-tests-packages.md)
- 01.3 [NumPy and vectorization: thinking in arrays](part-1-foundations/01-python-for-engineers/01.3-numpy-vectorization.md)
- 01.4 [Performance, profiling and concurrency](part-1-foundations/01-python-for-engineers/01.4-performance-concurrency.md)

### Module 02: [Linear Algebra](part-1-foundations/02-linear-algebra/) (45 h)

Every model in this course is, at the bottom, a pile of matrix multiplications. Let's understand them.

**When you finish it, you can:**

- Manipulate vectors, norms, dot products and angles, and use them as similarity measures.
- Treat matrices as linear maps; reason about rank, null space, inverses and conditioning.
- Derive least squares as an orthogonal projection and solve it stably (QR, lstsq) instead of inverting matrices.
- Compute and interpret eigendecompositions and the SVD, and use the SVD for low-rank approximation.

**Lessons:**

- 02.1 [Vectors, dot products and norms](part-1-foundations/02-linear-algebra/02.1-vectors-norms.md)
- 02.2 [Matrices as functions: multiplication, rank, inverses](part-1-foundations/02-linear-algebra/02.2-matrices-as-maps.md)
- 02.3 [Least squares and projections](part-1-foundations/02-linear-algebra/02.3-least-squares.md)
- 02.4 [Eigendecomposition and the SVD](part-1-foundations/02-linear-algebra/02.4-eigen-svd.md)

### Module 03: [Calculus and Optimization](part-1-foundations/03-calculus-optimization/) (40 h)

Training a model is minimizing a function. This module is about how you minimize functions, and why it sometimes doesn't work.

**When you finish it, you can:**

- Compute derivatives, partial derivatives and gradients of the functions used in ML, including matrix expressions.
- Apply the chain rule on computational graphs, which is all backpropagation is.
- Use Jacobians, Hessians and Taylor expansions to reason about local behaviour and curvature.
- Implement gradient descent, reason about step sizes and convexity, and verify gradients numerically.
- Solve constrained problems with Lagrange multipliers and read KKT conditions.

**Lessons:**

- 03.1 [Derivatives, gradients and the chain rule](part-1-foundations/03-calculus-optimization/03.1-derivatives-gradients.md)
- 03.2 [Jacobians, Hessians and Taylor: the local picture](part-1-foundations/03-calculus-optimization/03.2-jacobians-hessians.md)
- 03.3 [Gradient descent and convexity](part-1-foundations/03-calculus-optimization/03.3-gradient-descent-convexity.md)
- 03.4 [Constrained optimization: Lagrange multipliers and KKT](part-1-foundations/03-calculus-optimization/03.4-constrained-optimization.md)

### Module 04: [Probability](part-1-foundations/04-probability/) (40 h)

Machine learning is applied probability wearing a hoodie. You need to speak it fluently.

**When you finish it, you can:**

- Work with discrete and continuous random variables, PMFs, PDFs and CDFs.
- Use the core distributions (Bernoulli, binomial, Poisson, categorical, Gaussian, exponential, beta, multivariate Gaussian) and know where each shows up.
- Apply conditional probability, independence and Bayes' rule, including base rate reasoning.
- Compute expectations, variances and covariances, and explain the law of large numbers and the CLT.
- Derive maximum likelihood and MAP estimators and connect them to loss functions and regularization.

**Lessons:**

- 04.1 [Random variables and distributions](part-1-foundations/04-probability/04.1-random-variables.md)
- 04.2 [Joint, conditional and Bayes](part-1-foundations/04-probability/04.2-conditional-bayes.md)
- 04.3 [Expectation, variance and the limit theorems](part-1-foundations/04-probability/04.3-expectation-limit-theorems.md)
- 04.4 [Maximum likelihood and MAP](part-1-foundations/04-probability/04.4-mle-map.md)

### Module 05: [Statistics and Experimentation](part-1-foundations/05-statistics-experimentation/) (40 h)

The part of data science that decides whether your "improvement" is real or just noise with a nice chart.

**When you finish it, you can:**

- Build estimators and confidence intervals, analytically and with the bootstrap.
- Run and interpret hypothesis tests without committing the classic p-value crimes.
- Design, size and analyze A/B tests, including power analysis, peeking and multiple comparisons.
- Reason about causality: confounders, randomization, DAGs, and basic observational methods.

**Lessons:**

- 05.1 [Estimation, confidence intervals and the bootstrap](part-1-foundations/05-statistics-experimentation/05.1-estimation-bootstrap.md)
- 05.2 [Hypothesis testing and how people abuse it](part-1-foundations/05-statistics-experimentation/05.2-hypothesis-testing.md)
- 05.3 [A/B testing: power, peeking and other crimes](part-1-foundations/05-statistics-experimentation/05.3-ab-testing.md)
- 05.4 [Causal inference for engineers](part-1-foundations/05-statistics-experimentation/05.4-causal-inference.md)

### Module 06: [Information Theory](part-1-foundations/06-information-theory/) (15 h)

Short module, huge payoff. Cross-entropy loss, KL divergence, perplexity and half of the generative modeling literature suddenly make sense.

**When you finish it, you can:**

- Compute entropy, cross-entropy and KL divergence and explain them as coding costs.
- Explain why classifiers and language models are trained with cross-entropy, and what perplexity measures.
- Use mutual information and understand the compression view of learning.

**Lessons:**

- 06.1 [Entropy, cross-entropy and KL divergence](part-1-foundations/06-information-theory/06.1-entropy-kl.md)
- 06.2 [Mutual information and learning as compression](part-1-foundations/06-information-theory/06.2-mutual-information-compression.md)

### Module 07: [Computer Systems for ML](part-1-foundations/07-computer-systems/) (35 h)

Your model runs on a real machine with real memory, real floating point and a real network. Ignore that and it will remind you, usually in production.

**When you finish it, you can:**

- Pick the right data structure and estimate the time and memory complexity of your code.
- Explain IEEE 754 floating point, avoid catastrophic cancellation and overflow, and implement stable log-sum-exp and softmax.
- Reason about caches, memory bandwidth, CPU vs GPU execution and why some operations are memory-bound.
- Work with processes, signals, environment variables, HTTP and REST APIs on Linux.

**Lessons:**

- 07.1 [Algorithms and data structures ML engineers actually use](part-1-foundations/07-computer-systems/07.1-algorithms-data-structures.md)
- 07.2 [Floating point and numerical stability](part-1-foundations/07-computer-systems/07.2-floating-point.md)
- 07.3 [The machine: memory hierarchy, CPUs and GPUs](part-1-foundations/07-computer-systems/07.3-the-machine.md)
- 07.4 [Networks, APIs and Linux processes](part-1-foundations/07-computer-systems/07.4-networks-apis-linux.md)

## Part II: Data Engineering

Models eat data, and somebody has to cook. This part turns you into that somebody: SQL, modeling, formats, distributed processing, orchestration, streaming, and keeping the whole thing secure and observable.

### Module 08: [SQL and Relational Databases](part-2-data-engineering/08-sql-relational/) (45 h)

SQL is fifty years old and it will outlive most of the frameworks you are excited about today.

**When you finish it, you can:**

- Write correct SQL for filtering, grouping, joins, subqueries and CTEs, including NULL semantics.
- Use window functions for rankings, running totals, sessionization and deduplication.
- Read query plans, design indexes and fix slow queries.
- Explain transactions, isolation levels, and OLTP vs OLAP workloads.

**Lessons:**

- 08.1 [The relational model and SQL fundamentals](part-2-data-engineering/08-sql-relational/08.1-relational-model-sql.md)
- 08.2 [Joins, aggregation and window functions](part-2-data-engineering/08-sql-relational/08.2-joins-windows.md)
- 08.3 [Query plans, indexes and performance](part-2-data-engineering/08-sql-relational/08.3-query-plans-indexes.md)
- 08.4 [Transactions, isolation and OLTP vs OLAP](part-2-data-engineering/08-sql-relational/08.4-transactions-oltp-olap.md)

### Module 09: [Data Modeling](part-2-data-engineering/09-data-modeling/) (20 h)

The schema is the API of your data. Design it badly and every downstream team pays interest forever.

**When you finish it, you can:**

- Normalize a schema to 3NF and explain when to denormalize.
- Design star schemas with facts, dimensions, grain and slowly changing dimensions.
- Choose between dimensional models, wide tables and data vault for a given use case.

**Lessons:**

- 09.1 [Normalization and entity modeling](part-2-data-engineering/09-data-modeling/09.1-normalization.md)
- 09.2 [Dimensional modeling: facts, dimensions and SCDs](part-2-data-engineering/09-data-modeling/09.2-dimensional-modeling.md)

### Module 10: [Data Wrangling and Analysis](part-2-data-engineering/10-data-wrangling/) (40 h)

Eighty percent of the job, zero percent of the glamour.

**When you finish it, you can:**

- Use pandas efficiently: indexing, groupby, merge, reshape, dtypes, and avoiding the classic traps.
- Use Polars and DuckDB for fast single-node analytics on data bigger than RAM.
- Profile and clean data, and encode data quality expectations as tests.
- Run exploratory analyses and build honest visualizations that communicate a result.

**Lessons:**

- 10.1 [pandas without the pain](part-2-data-engineering/10-data-wrangling/10.1-pandas.md)
- 10.2 [Polars and DuckDB: the modern single-node stack](part-2-data-engineering/10-data-wrangling/10.2-polars-duckdb.md)
- 10.3 [Data cleaning and data quality](part-2-data-engineering/10-data-wrangling/10.3-cleaning-quality.md)
- 10.4 [Exploratory analysis and visualization that says something](part-2-data-engineering/10-data-wrangling/10.4-eda-visualization.md)

### Module 11: [Storage, Formats and Architecture](part-2-data-engineering/11-storage-formats/) (20 h)

Where the bytes live, what shape they have, and why that decides your cloud bill.

**When you finish it, you can:**

- Choose between row and columnar formats (CSV, JSON, Avro, Parquet, Arrow) and explain encodings, compression and predicate pushdown.
- Design object storage layouts with partitioning, and avoid the small files problem.
- Explain data lakes, warehouses and lakehouses, table formats like Iceberg and Delta, and the medallion architecture.

**Lessons:**

- 11.1 [File formats: CSV, JSON, Parquet, Avro, Arrow](part-2-data-engineering/11-storage-formats/11.1-file-formats.md)
- 11.2 [Lakes, warehouses and lakehouses](part-2-data-engineering/11-storage-formats/11.2-lakes-warehouses-lakehouses.md)

### Module 12: [Distributed Processing](part-2-data-engineering/12-distributed-processing/) (25 h)

When one machine is not enough, and how to tell whether it really isn't.

**When you finish it, you can:**

- Explain the MapReduce model, partitioning, shuffles and the Spark execution model (DAGs, stages, tasks).
- Write PySpark DataFrame jobs and diagnose skew, spills and bad joins.
- Decide when distributed processing is justified and when DuckDB on one box wins.

**Lessons:**

- 12.1 [From MapReduce to Spark: the distributed model](part-2-data-engineering/12-distributed-processing/12.1-mapreduce-spark-model.md)
- 12.2 [Spark in practice: partitions, shuffles, skew](part-2-data-engineering/12-distributed-processing/12.2-spark-in-practice.md)

### Module 13: [Pipelines and Orchestration](part-2-data-engineering/13-pipelines-orchestration/) (30 h)

A pipeline that only works when you run it by hand is a hobby, not a pipeline.

**When you finish it, you can:**

- Design ETL/ELT pipelines that are idempotent, incremental and backfillable.
- Orchestrate pipelines with Airflow or Dagster, including retries, sensors, SLAs and alerting.
- Build tested transformation layers with dbt.

**Lessons:**

- 13.1 [ETL, ELT, idempotency and backfills](part-2-data-engineering/13-pipelines-orchestration/13.1-etl-elt-idempotency.md)
- 13.2 [Orchestration with Airflow and Dagster](part-2-data-engineering/13-pipelines-orchestration/13.2-orchestration.md)
- 13.3 [Transformations and tests with dbt](part-2-data-engineering/13-pipelines-orchestration/13.3-dbt.md)

### Module 14: [Streaming](part-2-data-engineering/14-streaming/) (25 h)

Data that doesn't wait for your nightly batch job.

**When you finish it, you can:**

- Explain logs, topics, partitions, consumer groups and offsets in Kafka.
- Reason about event time vs processing time, windows, watermarks, state and delivery guarantees.
- Build a small streaming pipeline with change data capture.

**Lessons:**

- 14.1 [Logs, Kafka and event-driven architecture](part-2-data-engineering/14-streaming/14.1-kafka-event-driven.md)
- 14.2 [Stream processing: time, windows, state and exactly-once](part-2-data-engineering/14-streaming/14.2-stream-processing.md)

### Module 15: [Governance, Security and Reliability](part-2-data-engineering/15-governance-security/) (25 h)

Your data platform is the most valuable target in the company. Treat it that way.

**When you finish it, you can:**

- Threat-model a data platform: IAM, least privilege, encryption, secrets, network boundaries.
- Handle PII correctly: classification, masking, retention, and the basics of GDPR-style obligations.
- Use catalogs, lineage and data contracts to keep producers and consumers honest.
- Run data observability: freshness, volume, schema and distribution checks, alerting and on-call.

**Lessons:**

- 15.1 [Securing data platforms](part-2-data-engineering/15-governance-security/15.1-securing-data-platforms.md)
- 15.2 [Privacy, lineage, catalogs and data contracts](part-2-data-engineering/15-governance-security/15.2-privacy-lineage-contracts.md)
- 15.3 [Data observability and on-call](part-2-data-engineering/15-governance-security/15.3-observability-oncall.md)

## Part III: Classical Machine Learning

Statistical learning, derived and implemented from scratch, then used properly with scikit-learn. Most ML in production is still this, and anybody who tells you otherwise has never had to explain a model to an auditor.

### Module 16: [The Learning Problem](part-3-classical-ml/16-learning-problem/) (25 h)

Before algorithms: what does it even mean to learn from data, and when can you trust it?

**When you finish it, you can:**

- Frame a problem as supervised, unsupervised, or not-ML-at-all, with a loss and a baseline.
- Explain empirical risk minimization, generalization error, bias, variance and model capacity.
- Explain PAC learning, VC dimension, double descent and the curse of dimensionality at a working level.

**Lessons:**

- 16.1 [Framing: what learning actually is](part-3-classical-ml/16-learning-problem/16.1-framing.md)
- 16.2 [Generalization, bias and variance](part-3-classical-ml/16-learning-problem/16.2-generalization-bias-variance.md)
- 16.3 [Learning theory: PAC, VC and the curse of dimensionality](part-3-classical-ml/16-learning-problem/16.3-learning-theory.md)

### Module 17: [Linear Models](part-3-classical-ml/17-linear-models/) (35 h)

The most underrated models in the business. Also the gateway to neural networks.

**When you finish it, you can:**

- Derive linear regression as least squares, as a projection and as maximum likelihood, and implement it.
- Use ridge, lasso and elastic net, and explain them as priors and as constraints.
- Derive and implement logistic regression and softmax regression with gradient descent.
- Explain generalized linear models and choose link functions and losses for counts and rates.

**Lessons:**

- 17.1 [Linear regression three ways](part-3-classical-ml/17-linear-models/17.1-linear-regression.md)
- 17.2 [Regularization: ridge, lasso, elastic net](part-3-classical-ml/17-linear-models/17.2-regularization.md)
- 17.3 [Logistic regression and GLMs](part-3-classical-ml/17-linear-models/17.3-logistic-regression-glms.md)
- 17.4 [Multiclass, softmax and the road to neural nets](part-3-classical-ml/17-linear-models/17.4-softmax-regression.md)

### Module 18: [Evaluation](part-3-classical-ml/18-evaluation/) (30 h)

The module that protects you from your own optimism.

**When you finish it, you can:**

- Design train/validation/test splits and cross-validation for i.i.d., grouped and temporal data.
- Choose and compute metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, MAE, RMSE, MAPE, R².
- Calibrate probabilities, choose thresholds from business costs, and read reliability diagrams.
- Detect and prevent leakage, and deal with class imbalance without fooling yourself.

**Lessons:**

- 18.1 [Splits and cross-validation that don't lie](part-3-classical-ml/18-evaluation/18.1-splits-cross-validation.md)
- 18.2 [Classification and regression metrics](part-3-classical-ml/18-evaluation/18.2-metrics.md)
- 18.3 [Calibration, thresholds and costs](part-3-classical-ml/18-evaluation/18.3-calibration-thresholds.md)
- 18.4 [Leakage, imbalance and other ways to fool yourself](part-3-classical-ml/18-evaluation/18.4-leakage-imbalance.md)

### Module 19: [Instance, Probabilistic and Kernel Methods](part-3-classical-ml/19-instance-kernel-probabilistic/) (25 h)

Three families of classic models, each with an idea you will see again in deep learning.

**When you finish it, you can:**

- Implement k-nearest neighbors, pick distance metrics and explain its failure in high dimensions.
- Derive Naive Bayes and Gaussian discriminant analysis, and contrast generative and discriminative models.
- Derive the SVM primal and dual, explain the kernel trick and tune C and gamma.

**Lessons:**

- 19.1 [k-Nearest neighbors and distances](part-3-classical-ml/19-instance-kernel-probabilistic/19.1-knn.md)
- 19.2 [Naive Bayes and generative classifiers](part-3-classical-ml/19-instance-kernel-probabilistic/19.2-naive-bayes-generative.md)
- 19.3 [Support vector machines and kernels](part-3-classical-ml/19-instance-kernel-probabilistic/19.3-svm-kernels.md)

### Module 20: [Trees and Ensembles](part-3-classical-ml/20-trees-ensembles/) (35 h)

If your data lives in a table, this is probably the module that pays your salary.

**When you finish it, you can:**

- Implement a decision tree with Gini/entropy splits and explain pruning.
- Explain why bagging reduces variance and how random forests decorrelate trees.
- Derive gradient boosting as gradient descent in function space and implement it.
- Tune XGBoost, LightGBM and CatBoost, and handle categorical features, missing values and early stopping.

**Lessons:**

- 20.1 [Decision trees](part-3-classical-ml/20-trees-ensembles/20.1-decision-trees.md)
- 20.2 [Bagging and random forests](part-3-classical-ml/20-trees-ensembles/20.2-bagging-random-forests.md)
- 20.3 [Boosting: from AdaBoost to gradient boosting](part-3-classical-ml/20-trees-ensembles/20.3-boosting.md)
- 20.4 [XGBoost, LightGBM and CatBoost in the real world](part-3-classical-ml/20-trees-ensembles/20.4-gbdt-in-practice.md)

### Module 21: [Unsupervised Learning](part-3-classical-ml/21-unsupervised/) (30 h)

Finding structure when nobody gave you labels. Also finding structure that isn't there, if you're careless.

**When you finish it, you can:**

- Implement k-means and k-means++, and use hierarchical clustering and DBSCAN/HDBSCAN appropriately.
- Derive EM for Gaussian mixture models.
- Derive PCA from variance maximization and from the SVD, and use it correctly.
- Use t-SNE and UMAP for visualization without over-interpreting the pictures.

**Lessons:**

- 21.1 [Clustering: k-means, hierarchical, DBSCAN](part-3-classical-ml/21-unsupervised/21.1-clustering.md)
- 21.2 [Gaussian mixtures and EM](part-3-classical-ml/21-unsupervised/21.2-gmm-em.md)
- 21.3 [PCA and linear dimensionality reduction](part-3-classical-ml/21-unsupervised/21.3-pca.md)
- 21.4 [Manifold learning: t-SNE and UMAP](part-3-classical-ml/21-unsupervised/21.4-manifold-learning.md)

### Module 22: [The Practical Workflow](part-3-classical-ml/22-practical-workflow/) (25 h)

How the pieces fit together in a real project.

**When you finish it, you can:**

- Engineer features for numeric, categorical, text and datetime data inside leak-free pipelines.
- Tune hyperparameters with random search and Bayesian optimization (Optuna), with proper nested evaluation.
- Explain models with permutation importance, partial dependence and SHAP, and know the limits of each.

**Lessons:**

- 22.1 [Feature engineering and pipelines](part-3-classical-ml/22-practical-workflow/22.1-feature-engineering.md)
- 22.2 [Hyperparameter optimization](part-3-classical-ml/22-practical-workflow/22.2-hyperparameter-optimization.md)
- 22.3 [Interpretability: permutation importance, PDP, SHAP](part-3-classical-ml/22-practical-workflow/22.3-interpretability.md)

### Module 23: [Applied Domains](part-3-classical-ml/23-applied-domains/) (40 h)

Four kinds of problems that show up in real jobs constantly and in toy courses almost never.

**When you finish it, you can:**

- Forecast time series with baselines, ETS/ARIMA and gradient boosting, and backtest correctly.
- Build recommender systems with collaborative filtering, matrix factorization and two-tower retrieval.
- Detect anomalies with statistical methods, isolation forests and reconstruction error.
- Use Bayesian inference and Gaussian processes, and quantify uncertainty.

**Lessons:**

- 23.1 [Time series forecasting](part-3-classical-ml/23-applied-domains/23.1-time-series.md)
- 23.2 [Recommender systems](part-3-classical-ml/23-applied-domains/23.2-recommender-systems.md)
- 23.3 [Anomaly detection](part-3-classical-ml/23-applied-domains/23.3-anomaly-detection.md)
- 23.4 [Bayesian methods and Gaussian processes](part-3-classical-ml/23-applied-domains/23.4-bayesian-gaussian-processes.md)

## Part IV: Deep Learning

From a single neuron to transformers, diffusion models and multi-GPU training. You will build an autograd engine yourself before you are allowed to touch PyTorch, because after that PyTorch stops being magic.

### Module 24: [Neural Networks from Scratch](part-4-deep-learning/24-neural-nets-from-scratch/) (30 h)

No frameworks. Just NumPy, the chain rule and stubbornness.

**When you finish it, you can:**

- Explain the MLP, activation functions and universal approximation.
- Derive backpropagation for an MLP in matrix form.
- Build a scalar and a tensor autograd engine and train a network with it.

**Lessons:**

- 24.1 [From neurons to multilayer perceptrons](part-4-deep-learning/24-neural-nets-from-scratch/24.1-neurons-to-mlps.md)
- 24.2 [Backpropagation, derived properly](part-4-deep-learning/24-neural-nets-from-scratch/24.2-backpropagation.md)
- 24.3 [Building an autograd engine](part-4-deep-learning/24-neural-nets-from-scratch/24.3-autograd-engine.md)

### Module 25: [PyTorch](part-4-deep-learning/25-pytorch/) (25 h)

The framework, properly. Not the tutorial, the framework.

**When you finish it, you can:**

- Use tensors, autograd, views, broadcasting and devices without surprises.
- Write nn.Modules, Datasets, DataLoaders and a clean training loop with evaluation and checkpointing.
- Use GPUs, mixed precision and torch.compile, and make runs reproducible.

**Lessons:**

- 25.1 [Tensors and autograd](part-4-deep-learning/25-pytorch/25.1-tensors-autograd.md)
- 25.2 [Modules, data loading and the training loop](part-4-deep-learning/25-pytorch/25.2-modules-training-loop.md)
- 25.3 [GPUs, mixed precision and reproducibility](part-4-deep-learning/25-pytorch/25.3-gpus-mixed-precision.md)

### Module 26: [Training Deep Networks](part-4-deep-learning/26-training-deep-networks/) (40 h)

Why the same architecture trains beautifully for one person and diverges for another.

**When you finish it, you can:**

- Implement and compare SGD, momentum, Nesterov, RMSProp, Adam and AdamW.
- Choose initializations and learning-rate schedules (warmup, cosine, one-cycle) and read training curves.
- Regularize with weight decay, dropout, data augmentation, label smoothing and early stopping.
- Explain BatchNorm, LayerNorm, RMSNorm and residual connections, and why they make deep nets trainable.
- Debug training systematically: overfit one batch, check gradients, check data, check the loss.

**Lessons:**

- 26.1 [Optimizers: SGD, momentum, Adam, AdamW](part-4-deep-learning/26-training-deep-networks/26.1-optimizers.md)
- 26.2 [Initialization, learning rate schedules and training dynamics](part-4-deep-learning/26-training-deep-networks/26.2-init-schedules-dynamics.md)
- 26.3 [Regularization: weight decay, dropout, augmentation](part-4-deep-learning/26-training-deep-networks/26.3-regularization-dl.md)
- 26.4 [Normalization and residual connections](part-4-deep-learning/26-training-deep-networks/26.4-normalization-residuals.md)
- 26.5 [A recipe for training and debugging neural networks](part-4-deep-learning/26-training-deep-networks/26.5-training-recipe.md)

### Module 27: [Computer Vision](part-4-deep-learning/27-computer-vision/) (45 h)

Pixels in, decisions out.

**When you finish it, you can:**

- Implement convolution and pooling and compute receptive fields, output shapes and parameter counts.
- Explain the evolution of CNN architectures from LeNet to ResNet to ConvNeXt.
- Fine-tune pretrained models with transfer learning.
- Build and evaluate object detectors (IoU, NMS, mAP; two-stage, one-stage, DETR).
- Build segmentation models (U-Net, Mask R-CNN) and use promptable segmentation (SAM).

**Lessons:**

- 27.1 [Convolutions](part-4-deep-learning/27-computer-vision/27.1-convolutions.md)
- 27.2 [CNN architectures: LeNet to ConvNeXt](part-4-deep-learning/27-computer-vision/27.2-cnn-architectures.md)
- 27.3 [Transfer learning](part-4-deep-learning/27-computer-vision/27.3-transfer-learning.md)
- 27.4 [Object detection](part-4-deep-learning/27-computer-vision/27.4-object-detection.md)
- 27.5 [Segmentation](part-4-deep-learning/27-computer-vision/27.5-segmentation.md)

### Module 28: [Sequence Models](part-4-deep-learning/28-sequence-models/) (20 h)

Before transformers ate the world. Still worth knowing, for history and for streaming problems.

**When you finish it, you can:**

- Implement an RNN and backpropagation through time; explain vanishing and exploding gradients.
- Explain LSTM and GRU gating and seq2seq with attention.

**Lessons:**

- 28.1 [Recurrent networks and backprop through time](part-4-deep-learning/28-sequence-models/28.1-rnns-bptt.md)
- 28.2 [LSTM, GRU and seq2seq with attention](part-4-deep-learning/28-sequence-models/28.2-lstm-gru-seq2seq.md)

### Module 29: [Transformers](part-4-deep-learning/29-transformers/) (35 h)

The architecture behind almost everything interesting since 2017, built from the ground up.

**When you finish it, you can:**

- Derive scaled dot-product attention and multi-head attention and implement them.
- Build a transformer block (pre-norm, residuals, MLP) from scratch and train it.
- Explain positional encodings (sinusoidal, learned, RoPE, ALiBi) and efficient attention (FlashAttention, sparse, linear).

**Lessons:**

- 29.1 [Attention from first principles](part-4-deep-learning/29-transformers/29.1-attention.md)
- 29.2 [The transformer block, built from scratch](part-4-deep-learning/29-transformers/29.2-transformer-block.md)
- 29.3 [Positions, efficiency and FlashAttention](part-4-deep-learning/29-transformers/29.3-positions-efficiency.md)

### Module 30: [Representation Learning](part-4-deep-learning/30-representation-learning/) (20 h)

Learning useful features without labels, and connecting images to text.

**When you finish it, you can:**

- Explain and implement contrastive learning (InfoNCE, SimCLR) and non-contrastive methods (BYOL, DINO).
- Explain vision transformers and CLIP, and use CLIP for zero-shot classification and retrieval.

**Lessons:**

- 30.1 [Self-supervised and contrastive learning](part-4-deep-learning/30-representation-learning/30.1-self-supervised.md)
- 30.2 [Vision transformers and CLIP](part-4-deep-learning/30-representation-learning/30.2-vit-clip.md)

### Module 31: [Generative Models](part-4-deep-learning/31-generative-models/) (40 h)

Teaching machines to make things, and understanding the math instead of just the pretty samples.

**When you finish it, you can:**

- Derive the ELBO and implement a VAE with the reparameterization trick.
- Explain the GAN minimax game, its failure modes, and the fixes (WGAN-GP, spectral norm).
- Derive and implement a DDPM diffusion model.
- Explain flow matching, latent diffusion, and classifier-free guidance.

**Lessons:**

- 31.1 [Autoencoders and VAEs](part-4-deep-learning/31-generative-models/31.1-autoencoders-vae.md)
- 31.2 [GANs](part-4-deep-learning/31-generative-models/31.2-gans.md)
- 31.3 [Diffusion models](part-4-deep-learning/31-generative-models/31.3-diffusion.md)
- 31.4 [Flow matching, latent diffusion and guidance](part-4-deep-learning/31-generative-models/31.4-flow-matching-latent-diffusion.md)

### Module 32: [Graph Neural Networks](part-4-deep-learning/32-graph-neural-networks/) (10 h)

For data that is a network, not a table or a grid.

**When you finish it, you can:**

- Explain message passing, GCN, GraphSAGE and GAT, and implement a GCN.

**Lessons:**

- 32.1 [Message passing and GNNs](part-4-deep-learning/32-graph-neural-networks/32.1-message-passing.md)

### Module 33: [Deep Learning at Scale](part-4-deep-learning/33-dl-at-scale/) (30 h)

What changes when the model doesn't fit on one GPU and the bill has six digits.

**When you finish it, you can:**

- Estimate FLOPs, memory (weights, gradients, optimizer state, activations) and arithmetic intensity for a training run.
- Explain and use data parallelism (DDP), sharded training (FSDP/ZeRO), tensor and pipeline parallelism.
- Profile training, use activation checkpointing, fused kernels and torch.compile.

**Lessons:**

- 33.1 [GPU arithmetic: memory, FLOPs and bandwidth](part-4-deep-learning/33-dl-at-scale/33.1-gpu-arithmetic.md)
- 33.2 [Distributed training: DDP, FSDP, tensor and pipeline parallelism](part-4-deep-learning/33-dl-at-scale/33.2-distributed-training.md)
- 33.3 [Profiling and making it fast](part-4-deep-learning/33-dl-at-scale/33.3-profiling-speed.md)

## Part V: Language Models

From n-grams to reasoning models. You will write a tokenizer and a GPT from scratch, then learn how the big ones are pretrained, served, fine-tuned and aligned.

### Module 34: [Classical NLP](part-5-language-models/34-classical-nlp/) (25 h)

The ideas that came before transformers, and still run half of the text processing in the world.

**When you finish it, you can:**

- Normalize and tokenize text; build n-gram language models with smoothing and measure perplexity.
- Build bag-of-words and TF-IDF classifiers, and explain sequence labeling (POS, NER) with HMMs and CRFs.
- Train and evaluate word2vec embeddings and explain GloVe.

**Lessons:**

- 34.1 [Text, tokens and n-gram language models](part-5-language-models/34-classical-nlp/34.1-ngram-lms.md)
- 34.2 [Bag of words, TF-IDF and sequence labeling](part-5-language-models/34-classical-nlp/34.2-bow-tfidf-sequence-labeling.md)
- 34.3 [Word embeddings: word2vec and GloVe](part-5-language-models/34-classical-nlp/34.3-word-embeddings.md)

### Module 35: [Building an LLM](part-5-language-models/35-building-an-llm/) (50 h)

The whole stack of a decoder-only language model, minus the hundred million dollars.

**When you finish it, you can:**

- Implement byte-level BPE tokenization.
- Implement and train a GPT-style decoder-only transformer.
- Explain pretraining data pipelines: crawling, extraction, filtering, deduplication, mixing, contamination.
- Use scaling laws to budget compute, parameters and tokens.
- Explain the modern LLM architecture: RoPE, RMSNorm, SwiGLU, GQA, mixture of experts.

**Lessons:**

- 35.1 [Tokenization: BPE from scratch](part-5-language-models/35-building-an-llm/35.1-bpe-tokenizer.md)
- 35.2 [A GPT from scratch](part-5-language-models/35-building-an-llm/35.2-gpt-from-scratch.md)
- 35.3 [Pretraining data](part-5-language-models/35-building-an-llm/35.3-pretraining-data.md)
- 35.4 [Scaling laws and compute budgets](part-5-language-models/35-building-an-llm/35.4-scaling-laws.md)
- 35.5 [The modern LLM architecture: RoPE, RMSNorm, SwiGLU, GQA, MoE](part-5-language-models/35-building-an-llm/35.5-modern-architecture.md)

### Module 36: [Encoders and Encoder-Decoders](part-5-language-models/36-encoders/) (15 h)

The models that quietly power search, classification and extraction everywhere.

**When you finish it, you can:**

- Explain masked language modeling (BERT) and span corruption (T5).
- Fine-tune an encoder for classification and token classification, and train sentence embeddings.

**Lessons:**

- 36.1 [BERT, T5 and fine-tuning for NLP tasks](part-5-language-models/36-encoders/36.1-bert-t5.md)

### Module 37: [Inference](part-5-language-models/37-inference/) (25 h)

Training happens once. Inference happens every time a user types something. That's where the money goes.

**When you finish it, you can:**

- Implement greedy, beam, temperature, top-k and top-p decoding, and constrained decoding.
- Explain the KV cache, continuous batching, PagedAttention and speculative decoding.
- Quantize models (int8, int4, GPTQ, AWQ, GGUF) and measure the quality cost.

**Lessons:**

- 37.1 [Decoding strategies](part-5-language-models/37-inference/37.1-decoding.md)
- 37.2 [KV cache, batching and serving engines](part-5-language-models/37-inference/37.2-kv-cache-serving.md)
- 37.3 [Quantization](part-5-language-models/37-inference/37.3-quantization.md)

### Module 38: [Post-Training](part-5-language-models/38-post-training/) (40 h)

How a text predictor becomes an assistant, and how assistants learned to reason.

**When you finish it, you can:**

- Run supervised fine-tuning with chat templates and loss masking.
- Implement LoRA and use QLoRA to fine-tune on consumer hardware.
- Explain RLHF (reward models, PPO with KL penalty) and derive and implement DPO.
- Explain reasoning models: chain of thought, RL with verifiable rewards, GRPO and test-time compute.

**Lessons:**

- 38.1 [Supervised fine-tuning and instruction tuning](part-5-language-models/38-post-training/38.1-sft.md)
- 38.2 [Parameter-efficient fine-tuning: LoRA and QLoRA](part-5-language-models/38-post-training/38.2-lora-qlora.md)
- 38.3 [Preference optimization: RLHF and DPO](part-5-language-models/38-post-training/38.3-rlhf-dpo.md)
- 38.4 [Reasoning models and RL with verifiable rewards](part-5-language-models/38-post-training/38.4-reasoning-rlvr.md)

### Module 39: [Evaluating and Extending LLMs](part-5-language-models/39-evaluating-extending-llms/) (20 h)

How to tell if a model is good, and how models learned to see and hear.

**When you finish it, you can:**

- Evaluate LLMs with benchmarks, held-out tasks, LLM-as-judge and human evaluation, and detect contamination.
- Explain vision-language models and speech models (Whisper-style) and how modalities are fused.

**Lessons:**

- 39.1 [Evaluating language models](part-5-language-models/39-evaluating-extending-llms/39.1-evaluating-llms.md)
- 39.2 [Multimodal models: vision-language and speech](part-5-language-models/39-evaluating-extending-llms/39.2-multimodal.md)

## Part VI: AI Engineering

Building products on top of foundation models: APIs, prompts, tools, retrieval, agents, evals, security and cost. This is where most new AI jobs are, and where most of the embarrassing incidents happen.

### Module 40: [Building on Foundation Models](part-6-ai-engineering/40-building-on-foundation-models/) (25 h)

The LLM is a component. Treat it like one.

**When you finish it, you can:**

- Call LLM APIs robustly: streaming, retries, timeouts, rate limits, token accounting and cost estimation.
- Engineer prompts systematically and get reliable structured outputs validated against schemas.
- Implement tool use / function calling loops safely.

**Lessons:**

- 40.1 [The LLM as an API: tokens, latency and money](part-6-ai-engineering/40-building-on-foundation-models/40.1-llm-as-api.md)
- 40.2 [Prompting as programming and structured outputs](part-6-ai-engineering/40-building-on-foundation-models/40.2-prompting-structured-outputs.md)
- 40.3 [Tool use and function calling](part-6-ai-engineering/40-building-on-foundation-models/40.3-tool-use.md)

### Module 41: [Retrieval](part-6-ai-engineering/41-retrieval/) (35 h)

Giving the model the right context at the right time. Harder than it sounds.

**When you finish it, you can:**

- Use embeddings for semantic search and pick an embedding model with evidence.
- Explain and tune approximate nearest neighbor indexes (HNSW, IVF, PQ).
- Build a RAG pipeline with chunking, hybrid search (BM25 + dense), reranking and citations.
- Evaluate retrieval (recall@k, MRR, nDCG) and generation (faithfulness), and use advanced patterns (query rewriting, HyDE, GraphRAG, agentic retrieval).

**Lessons:**

- 41.1 [Embeddings and semantic search](part-6-ai-engineering/41-retrieval/41.1-embeddings-semantic-search.md)
- 41.2 [Vector indexes: HNSW, IVF, PQ](part-6-ai-engineering/41-retrieval/41.2-vector-indexes.md)
- 41.3 [RAG done properly](part-6-ai-engineering/41-retrieval/41.3-rag.md)
- 41.4 [Advanced RAG and evaluating retrieval](part-6-ai-engineering/41-retrieval/41.4-advanced-rag-evaluation.md)

### Module 42: [Agents](part-6-ai-engineering/42-agents/) (30 h)

An LLM in a loop with tools. Capable, fragile, and a security team's nightmare.

**When you finish it, you can:**

- Build an agent loop (ReAct) with planning, memory and stopping conditions.
- Build and consume MCP servers and design tool interfaces for models.
- Design multi-agent and workflow systems, and know when a plain workflow beats an agent.

**Lessons:**

- 42.1 [The agent loop: ReAct, planning and memory](part-6-ai-engineering/42-agents/42.1-agent-loop.md)
- 42.2 [MCP and tool ecosystems](part-6-ai-engineering/42-agents/42.2-mcp-tools.md)
- 42.3 [Multi-agent systems and workflow design](part-6-ai-engineering/42-agents/42.3-multi-agent-workflows.md)

### Module 43: [Evals and Observability](part-6-ai-engineering/43-evals-observability/) (20 h)

If you can't measure it, you're doing vibes-based engineering.

**When you finish it, you can:**

- Build eval suites from real failures: golden sets, assertions, LLM-as-judge with calibration.
- Trace and monitor LLM applications: latency, cost, quality, feedback and regressions.

**Lessons:**

- 43.1 [Building eval suites](part-6-ai-engineering/43-evals-observability/43.1-eval-suites.md)
- 43.2 [Tracing, monitoring and feedback loops](part-6-ai-engineering/43-evals-observability/43.2-tracing-monitoring.md)

### Module 44: [LLM Security](part-6-ai-engineering/44-llm-security/) (20 h)

My favourite module. Everything you build in Part VI, we now break.

**When you finish it, you can:**

- Explain and demonstrate direct and indirect prompt injection, jailbreaks and data exfiltration.
- Apply the OWASP Top 10 for LLM applications to a real design.
- Red-team an LLM application and build layered defenses and guardrails.

**Lessons:**

- 44.1 [Prompt injection and the OWASP LLM Top 10](part-6-ai-engineering/44-llm-security/44.1-prompt-injection-owasp.md)
- 44.2 [Red teaming and guardrails](part-6-ai-engineering/44-llm-security/44.2-red-teaming-guardrails.md)

### Module 45: [LLM Systems in Production](part-6-ai-engineering/45-llm-production/) (20 h)

Running this stuff for real users without going bankrupt.

**When you finish it, you can:**

- Design LLM gateways with caching (exact and semantic), routing, fallbacks and budgets.
- Self-host open-weight models with vLLM or llama.cpp and size the hardware.

**Lessons:**

- 45.1 [Gateways, caching, routing and cost control](part-6-ai-engineering/45-llm-production/45.1-gateways-caching-routing.md)
- 45.2 [Self-hosting open models](part-6-ai-engineering/45-llm-production/45.2-self-hosting.md)

## Part VII: Reinforcement Learning

Learning from interaction instead of labels. Required for RLHF and reasoning models, useful for a lot more.

### Module 46: [Reinforcement Learning](part-7-reinforcement-learning/46-reinforcement-learning/) (50 h)

Sutton and Barto, condensed, implemented, and connected to the LLM world.

**When you finish it, you can:**

- Solve multi-armed bandits with epsilon-greedy, UCB and Thompson sampling.
- Formulate MDPs, derive the Bellman equations and solve them with value and policy iteration.
- Implement Monte Carlo, SARSA and Q-learning.
- Implement DQN with replay buffers and target networks.
- Derive the policy gradient theorem and implement REINFORCE, actor-critic and PPO.
- Explain MCTS and AlphaZero, and connect PPO/GRPO to LLM post-training.

**Lessons:**

- 46.1 [Bandits and exploration](part-7-reinforcement-learning/46-reinforcement-learning/46.1-bandits.md)
- 46.2 [MDPs, Bellman equations and dynamic programming](part-7-reinforcement-learning/46-reinforcement-learning/46.2-mdps-bellman-dp.md)
- 46.3 [Monte Carlo and temporal difference: SARSA and Q-learning](part-7-reinforcement-learning/46-reinforcement-learning/46.3-mc-td.md)
- 46.4 [Deep Q-networks](part-7-reinforcement-learning/46-reinforcement-learning/46.4-dqn.md)
- 46.5 [Policy gradients and actor-critic](part-7-reinforcement-learning/46-reinforcement-learning/46.5-policy-gradients.md)
- 46.6 [PPO, search, and RL in the wild](part-7-reinforcement-learning/46-reinforcement-learning/46.6-ppo-search.md)

## Part VIII: MLOps and ML Systems

The difference between a notebook and a product. System design, reproducibility, serving, testing, deployment, monitoring, infrastructure, security and responsibility.

### Module 47: [ML System Design](part-8-mlops/47-ml-system-design/) (25 h)

Thinking about the whole system before you write the first line.

**When you finish it, you can:**

- Turn a business problem into an ML design doc: objective, metrics, data, baseline, architecture, risks.
- Design end-to-end ML systems for search, recommendations, fraud and LLM products.

**Lessons:**

- 47.1 [From business problem to ML design doc](part-8-mlops/47-ml-system-design/47.1-design-doc.md)
- 47.2 [Design case studies](part-8-mlops/47-ml-system-design/47.2-case-studies.md)

### Module 48: [Reproducibility](part-8-mlops/48-reproducibility/) (15 h)

"It worked on my machine" is not a deployment strategy.

**When you finish it, you can:**

- Track experiments and register models with MLflow.
- Version data and pipelines (DVC), pin environments and make training runs reproducible.

**Lessons:**

- 48.1 [Experiment tracking and model registries](part-8-mlops/48-reproducibility/48.1-tracking-registry.md)
- 48.2 [Data versioning, pipelines and environments](part-8-mlops/48-reproducibility/48.2-data-versioning.md)

### Module 49: [Serving](part-8-mlops/49-serving/) (30 h)

Getting predictions to the people who need them, fast and cheaply.

**When you finish it, you can:**

- Choose between batch, online and streaming inference and implement an online service with FastAPI.
- Containerize models with Docker and deploy them on Kubernetes with autoscaling.
- Optimize models for inference with ONNX, TensorRT-style compilation, distillation and pruning.

**Lessons:**

- 49.1 [Serving patterns: batch, online, streaming](part-8-mlops/49-serving/49.1-serving-patterns.md)
- 49.2 [Containers and Kubernetes for ML](part-8-mlops/49-serving/49.2-containers-kubernetes.md)
- 49.3 [Making models small and fast: ONNX, TensorRT, distillation, pruning](part-8-mlops/49-serving/49.3-small-fast-models.md)

### Module 50: [Testing, CI/CD and Deployment](part-8-mlops/50-testing-cicd-deployment/) (20 h)

Shipping models without holding your breath.

**When you finish it, you can:**

- Test ML code, data and models; build CI/CD/CT pipelines for ML.
- Deploy with shadow mode, canaries, A/B tests and bandits, and roll back safely.

**Lessons:**

- 50.1 [Testing ML systems and CI/CD/CT](part-8-mlops/50-testing-cicd-deployment/50.1-testing-cicd.md)
- 50.2 [Deployment strategies: shadow, canary, A/B, bandits](part-8-mlops/50-testing-cicd-deployment/50.2-deployment-strategies.md)

### Module 51: [Monitoring](part-8-mlops/51-monitoring/) (15 h)

Models rot. Monitoring is how you find out before the customers do.

**When you finish it, you can:**

- Detect data drift, prediction drift and concept drift with PSI, KS tests and performance monitoring.
- Design monitoring, alerting and retraining triggers, and handle delayed labels and feedback loops.

**Lessons:**

- 51.1 [Drift, data quality and model monitoring](part-8-mlops/51-monitoring/51.1-drift-monitoring.md)

### Module 52: [Cloud and Infrastructure](part-8-mlops/52-cloud-infrastructure/) (15 h)

Somebody else's computer, with a bill.

**When you finish it, you can:**

- Use the core cloud primitives (compute, object storage, IAM, networking, managed ML) and infrastructure as code.
- Choose GPU instances, estimate costs and avoid the classic cloud bill disasters.

**Lessons:**

- 52.1 [Cloud, GPUs, infrastructure as code and cost](part-8-mlops/52-cloud-infrastructure/52.1-cloud-iac-cost.md)

### Module 53: [Security of ML Systems](part-8-mlops/53-ml-security/) (20 h)

The attack surface nobody put in the architecture diagram.

**When you finish it, you can:**

- Implement and defend against evasion attacks (FGSM, PGD), poisoning and backdoors, model extraction and membership inference.
- Secure the ML supply chain: unsafe serialization (pickle), model hubs, dependencies, SBOMs and signing.

**Lessons:**

- 53.1 [Adversarial machine learning](part-8-mlops/53-ml-security/53.1-adversarial-ml.md)
- 53.2 [The ML supply chain](part-8-mlops/53-ml-security/53.2-ml-supply-chain.md)

### Module 54: [Responsible AI](part-8-mlops/54-responsible-ai/) (25 h)

Not a compliance checkbox. The part where your model meets actual human beings.

**When you finish it, you can:**

- Measure and mitigate unfairness with group metrics, and explain the impossibility results.
- Apply differential privacy (DP-SGD) and federated learning, and know their costs.
- Document models and data (model cards, datasheets) and map systems to regulation (EU AI Act, NIST AI RMF).

**Lessons:**

- 54.1 [Fairness](part-8-mlops/54-responsible-ai/54.1-fairness.md)
- 54.2 [Privacy: differential privacy and federated learning](part-8-mlops/54-responsible-ai/54.2-privacy-dp-federated.md)
- 54.3 [Governance, regulation and documentation](part-8-mlops/54-responsible-ai/54.3-governance-regulation.md)

## Part IX: Capstones and Career

Four projects that prove you can do the job, and some honest advice on getting paid for it.

### Module 55: [Capstones](part-9-capstones/55-capstones/) (240 h)

No new theory. Just you, a spec, and the whole course behind you.

**When you finish it, you can:**

- Build and operate an end-to-end data platform.
- Ship, monitor and retrain a tabular ML product.
- Pretrain and post-train a small language model and evaluate it honestly.
- Build a production RAG agent, then attack it and fix what you find.

**Lessons:**

- 55.1 [Capstone A: the data platform](part-9-capstones/55-capstones/55.1-capstone-a-data-platform.md)
- 55.2 [Capstone B: the tabular ML product](part-9-capstones/55-capstones/55.2-capstone-b-ml-product.md)
- 55.3 [Capstone C: train and post-train a small language model](part-9-capstones/55-capstones/55.3-capstone-c-small-lm.md)
- 55.4 [Capstone D: a production RAG agent, attacked by you](part-9-capstones/55-capstones/55.4-capstone-d-rag-agent.md)

### Module 56: [Career](part-9-capstones/56-career/) (10 h)

Getting hired, and not becoming obsolete afterwards.

**When you finish it, you can:**

- Build a portfolio that shows engineering, not tutorials.
- Pass coding, ML theory and ML system design interviews.
- Keep up with the field by reading papers efficiently and ignoring the noise.

**Lessons:**

- 56.1 [Portfolio, interviews and ML system design interviews](part-9-capstones/56-career/56.1-portfolio-interviews.md)
- 56.2 [Staying current without drowning](part-9-capstones/56-career/56.2-staying-current.md)
