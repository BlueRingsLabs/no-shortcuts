# No Shortcuts

### Data, Machine Learning, Deep Learning and AI Engineering, the long way round

*by Gonzalo L. Romero (DeepRat)*

<p align="center">
  <img src="assets/cover.jpg" alt="An orange octopus at a desk late at night, surrounded by books on machine learning, deep learning and security, a laptop, a notebook with a reinforcement learning diagram, and a mug that says learn, code, repeat" width="480">
</p>

---

Every week somebody on the internet promises to turn you into an "AI Engineer" in six weeks. You watch the videos, you copy the notebook, you call `model.fit()`, you get 97% accuracy on a dataset that has been solved since before you were born, and you put "AI" on your CV. Then you get your first real job, somebody hands you a table with 400 million rows, a label that is wrong 8% of the time, a latency budget of 50 ms and a security team that wants to know why your chatbot just emailed the customer database to a stranger. And the six-week course has nothing to say about any of that.

This course is my answer to that problem. It is long. It is not easy. It has math in it, and the math is there because you need it, not because I want to look smart. It has a lot of code, and every piece of code in the labs is executed by CI before it lands here, because I don't trust code I haven't run, and neither should you.

I'm a systems engineer (technically still a student, the paperwork is slower than the learning), a certified AI engineer, and I've been writing software for more than fifteen years: AI systems that had to work outside of a demo, data platforms that had to be secure because the data was worth stealing, and a fair amount of penetration testing, which is a polite way of saying I get paid to find out how other people's systems break. That last part shows up all over this course. When I teach you RAG, I will also teach you how to attack it. When I teach you data pipelines, I will show you where the credentials leak. An engineer who doesn't know how their system fails is not an engineer yet. They are an optimist.

## What "completing this course" is supposed to mean

When you finish, you should be able to walk into a job as a **Data Engineer**, **Data Scientist**, **Machine Learning Engineer**, **Deep Learning Engineer** or **AI Engineer** and do the work. That means more than knowing what the words mean:

- You can take a messy business question and turn it into a well-posed learning problem, or tell people honestly that it isn't one.
- You can build the data platform under it: ingestion, storage, modeling, orchestration, streaming, quality checks, access control.
- You can derive, implement from scratch and then use in production the classical ML toolbox, and you know why gradient boosting still wins on tabular data.
- You can build and train deep networks in PyTorch, including a transformer and a small GPT, and you understand what is happening on the GPU while it trains.
- You can pretrain, fine-tune, align, quantize and serve language models, and you can build RAG systems and agents on top of them that survive contact with real users and real attackers.
- You can put all of the above into production with tracking, testing, CI/CD, monitoring, rollbacks, cost control, and a threat model.
- You know the responsibilities that come with it: fairness, privacy, regulation, and not shipping garbage because a deadline said so.

The full, boring, explicit version of that promise lives in [COMPETENCIES.md](COMPETENCIES.md). It maps every skill that the five roles require to the exact lesson that teaches it and the lab or capstone that proves you have it. If a skill is not on that map, I either consider it out of scope (and say so) or I missed it, in which case open an issue and tell me.

## How it's organized

Ten parts, 57 modules, 173 lessons. The full catalogue is in [SYLLABUS.md](SYLLABUS.md).

| Part | What it covers | Why it's there |
|---|---|---|
| [0. Orientation](part-0-orientation/) | The job titles, your workstation, how to study | So you don't waste the first three months |
| [I. Foundations](part-1-foundations/) | Python, linear algebra, calculus, probability, statistics, information theory, systems | Everything else is built on this. No, you can't skip it |
| [II. Data Engineering](part-2-data-engineering/) | SQL, modeling, wrangling, formats, Spark, orchestration, streaming, governance | Models eat data. Somebody has to cook |
| [III. Classical Machine Learning](part-3-classical-ml/) | Learning theory, linear models, evaluation, SVMs, trees, boosting, unsupervised, forecasting, recsys | Most production ML is still this, and it will stay that way |
| [IV. Deep Learning](part-4-deep-learning/) | Backprop from scratch, PyTorch, training, CNNs, RNNs, transformers, generative models, GNNs, scale | The engine under modern AI |
| [V. Language Models](part-5-language-models/) | Classical NLP, tokenizers, GPT from scratch, scaling, inference, fine-tuning, alignment, reasoning | How the thing everybody talks about actually works |
| [VI. AI Engineering](part-6-ai-engineering/) | LLM APIs, prompting, tools, retrieval, RAG, agents, MCP, evals, LLM security | Building products on top of foundation models without embarrassing yourself |
| [VII. Reinforcement Learning](part-7-reinforcement-learning/) | Bandits, MDPs, Q-learning, DQN, policy gradients, PPO | Needed for RLHF, reasoning models, and a few other things |
| [VIII. MLOps and ML Systems](part-8-mlops/) | System design, tracking, serving, CI/CD, monitoring, cloud, ML security, responsible AI | The difference between a notebook and a product |
| [IX. Capstones and Career](part-9-capstones/) | Four end-to-end projects and how to get hired | Proof |

Every lesson has the same bones: the idea, the math when there is math, code, the ways it breaks in production, exercises with answers hidden behind a click, and a short list of primary sources. I link to papers and real textbooks, not to blog posts that summarize blog posts.

## How long this takes

Honestly? If you start with decent programming skills and high school math, about **1,800 hours**, and that's my estimate for someone who actually does the labs. At 15 hours a week that's a bit over two years. At full time, about eleven months. Roughly the workload of a serious master's degree, minus the tuition and the group projects where one person does everything. Anybody who tells you it can be done in a weekend is selling something, usually a weekend course.

You can go faster through parts you already know. Every module starts with a short self-check. If you can do the self-check without looking anything up, skip the module and go do its lab. If the lab is easy, you really can skip it. If the lab is not easy, you were lying to the self-check. Happens to everyone :)

## Prerequisites

- You can program in *some* language and you're not scared of a terminal. Python specifically is taught in Part I, but I won't teach you what a loop is.
- High school algebra. Everything above that (linear algebra, calculus, probability, statistics) is taught from the ground up in Part I, fast but properly.
- A computer with 16 GB of RAM is comfortable. 8 GB works for almost everything. A GPU is nice but not required until Part IV, and even there every lab has a CPU-sized version. Free Colab or Kaggle GPUs cover the rest.

## Running the labs

Clone the repository (GitHub's green **Code** button gives you the URL) or download it, then, from its folder:

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -r requirements.txt        # or pip install -r requirements.txt
python tools/run_labs.py --part 1         # run every lab in Part I
python part-1-foundations/02-linear-algebra/labs/lab_02_3_least_squares.py
```

Labs are plain Python files split into cells with `# %%` markers. VS Code, PyCharm and Jupyter (via `jupytext`) all understand that format and will let you run them cell by cell. I don't keep notebooks as the source of truth because notebooks are where reproducibility goes to die: hidden state, out-of-order execution, diffs that nobody can review. If you prefer notebooks, `jupytext --to notebook lab.py` gives you one in a second. Just don't commit it back.

Every lab ends with assertions. If an assertion fails, the lab is telling you that you broke something. Read the message.

The numbers quoted in the lessons come from running the labs on a four-core CPU. Yours will differ in the last digit or two, because floating point on a different CPU, thread count or library version rounds differently, and several labs train small models. The assertions check the conclusions (this method beats that one, this error rate falls below that one), not the digits. If an assertion fails on your machine and you didn't touch the code, that's a bug in the lab, and I want to hear about it.

Labs from Part IV onward that train on this course's own text, or search it, read a frozen copy of Parts 0 to V (`part-5-language-models/_shared/course_snapshot.json.gz`), not the live lessons, so that a typo fix in a lesson doesn't retrain the shared models and change every number that depends on them. The small models themselves are trained on first use and cached in `~/.cache/no-shortcuts` (or `$NO_SHORTCUTS_CACHE`); the first run of a Part V lab takes a few minutes longer than the rest.

## Status

[STATUS.md](STATUS.md) is generated from the repository and shows, module by module, how many lessons and labs exist. Finished means: lessons written, labs passing in CI, exercises answered.

## License and contributions

This course is free, and it stays free. It's licensed under [Creative Commons Attribution-NonCommercial 4.0](LICENSE) (CC BY-NC 4.0): copy it, share it, translate it, teach from it, build on it, as long as you credit me by name (Gonzalo L. Romero, DeepRat) with a link back here, say what you changed, and don't make money with it. No paid courses repackaging these lessons, no paywalled "bootcamps", no selling the labs. If you want to use it in a way that makes money, ask first. The [LICENSE](LICENSE) file has the legal text, which is what actually applies.

Found a bug, a wrong derivation, a broken link, a lab that doesn't run on your machine? Open an issue. I'd rather be corrected than be wrong in public for years. Pull requests that fix errors are welcome. Pull requests that "improve the tone" will be closed with a smile.

---

*"It's not magic. It's linear algebra, a lot of electricity, and somebody who read the error message."*
