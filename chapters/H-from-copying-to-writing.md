# H. From copying to writing

The miniatures give you the model of each library. This chapter is about turning that model into the thing you actually wanted at the start: writing code you've never seen, for a problem no tutorial covered. It's the most practical chapter of the addendum, and the least technical.

## 1. What changes when you have the model

Before, a new situation meant searching for an example that matched it closely enough to copy. With the data model and the conventions in your head, you can **predict**. You see `df.groupby("store").transform("mean")` for the first time and you know, before running it, that the result has one value per row, aligned on the original index, because that's what `transform` means in split-apply-combine. You see a new scikit-learn class and you know it takes hyperparameters in the constructor and data in `fit`, and that its learned state ends with an underscore. You see a tensor operation that fails and you look at shapes and strides first.

Prediction is also how you'll know you're not there yet. If you can't say what a line will do before running it, that line is still copied, even if you typed it.

## 2. Six habits that make the jump

These are the exercises I'd give anyone stuck in the copy-paste phase. None of them requires anything you don't have.

**1. Predict, then run.** Before running any cell, write down (a comment is enough) what you expect: the shape, the dtype, the first few values, or the error. Then run it. Keep a count of how often you're right. When the count is high for a library, you know it.

**2. Rewrite from memory, the next day.** Take something you wrote or studied yesterday (a miniature from this addendum, a training loop, a pipeline) and write it again without looking, on paper if you can stand it. Then type it and run it, or run the test against the real library. The parts you couldn't remember are the parts you never understood, and now you know exactly which ones they are.

**3. Translate between libraries.** Take a piece of pandas code and rewrite it in plain NumPy; take a scikit-learn model and rewrite its `fit` by hand; take a PyTorch training loop and rewrite it with the miniature from chapter F. Translating forces you to know what the code does, because you can't translate what you don't understand.

**4. Break it on purpose.** Change one thing, the kind of mistake you'd make on a tired day: transpose a matrix, drop `keepdims`, forget `zero_grad`, put a layer in a list, call `fit_transform` on the test set. Predict what breaks, then read what actually broke. You'll learn more about a library's error messages in an afternoon of breaking things than in a year of avoiding errors.

**5. Read the source of one function a week.** Pick a function you use often and don't fully trust: `train_test_split`, `StandardScaler.partial_fit`, `DataFrame.merge`, `torch.nn.functional.cross_entropy`. Read it with `inspect.getsource`, or on GitHub, until you can explain what it does with each argument. A year of that is fifty functions you know from the inside.

**6. Write a test for something you believe.** Whenever you catch yourself assuming something about a library ("this returns a copy", "this sorts the keys", "this ignores NaN"), write three lines that check it. Either way you end up with a test, and now and then with a bug you found before it found you.

## 3. Which libraries, in what order

For machine learning, deep learning and RAG, this is the order I'd recommend, and the reason for each step:

1. **Python itself**, well enough to read a class with `__init__`, `__getitem__` and `__call__` without slowing down. Every library in this addendum is built from those.
2. **NumPy.** Everything else sits on it, and [chapter C](C-numpy-in-miniature.md) is the part to know cold.
3. **pandas** (and, when your data stops fitting comfortably in memory, Polars or DuckDB). Real data comes in tables, and most of the work of ML is getting those tables right.
4. **matplotlib**, just enough to plot a histogram, a scatter plot and a learning curve. You can't debug data you don't look at.
5. **scikit-learn.** Classical models, but above all the habits: pipelines, cross-validation, honest evaluation. These carry over to everything else.
6. **PyTorch.** Tensors, autograd, modules, the training loop, data loading.
7. **Hugging Face `transformers` and `sentence-transformers`**, once you know what a transformer is. They're PyTorch modules with a lot of convenience around them, and [chapter G](G-sentence-transformers-in-miniature.md) shows what one of those conveniences does.
8. **A vector index** (FAISS, or the index inside a vector database) and an **LLM API**, for RAG. By then they'll be one more library with a data model and conventions, and you'll know how to learn that.

The order matters more than the speed. Jumping to step 7 without steps 2 and 6 is exactly how you end up copying `model.encode(...)` without knowing what comes back.

## 4. Projects that force you to use them

Each of these can be done on a laptop with no GPU, with data that ships with the libraries (scikit-learn includes several small datasets that load without downloading anything: `load_digits`, `load_breast_cancer`, `load_diabetes`, `load_wine`). Each has a "done when", because a project without a finish line becomes another tutorial.

1. **Linear regression three ways.** On `load_diabetes`: by hand with NumPy (the normal equations, or `np.linalg.lstsq`), with your `MiniLinearRegression`, and with scikit-learn. *Done when* the three give the same coefficients and you can explain why.
2. **A messy table, cleaned.** Take any public CSV that interests you (open government data is full of them), or make one messy on purpose: duplicated rows, a numeric column stored as text, missing values, a date in three formats. *Done when* you have a cleaning function, a set of `assert`s that check the result, and you've printed `.isna().mean()` before and after.
3. **An honest pipeline.** On `load_breast_cancer`: a `Pipeline` with scaling and a model, evaluated with cross-validation, plus a deliberately leaky version (scaling or feature selection outside the pipeline) next to it. *Done when* you can show the gap between the two, and the honest number is the one you report.
4. **Digits with PyTorch.** On `load_digits` (8×8 images): a small MLP with `nn.Module`, a `Dataset`, a `DataLoader` and a training loop you write without copying. Then the same with your miniature from chapter F. *Done when* the test accuracy is above 95%, and you can say what each line of the loop does.
5. **Search your own notes.** Take a folder of your own text (notes, documentation you like, your own blog posts), split it into paragraphs, embed them with sentence-transformers, and find the closest paragraphs to a question with a dot product in NumPy. *Done when* you've checked that your hand-written `encode` ([chapter G](G-sentence-transformers-in-miniature.md)) gives the same vectors, and you've found one query where the search fails and understood why.
6. **A tiny RAG.** Project 5 plus an LLM: retrieve the top three paragraphs and put them in the prompt. *Done when* you have ten questions with known answers and you measure how many it gets right, with and without retrieval.

> [!TIP]
> Put every project in a Git repository from the first day, with a README that says what it does and how to run it. In a year you'll have a portfolio without having tried to build one, and you'll be able to see how much your code changed.

## 5. A setup for a normal laptop

You don't need a powerful machine for anything in this addendum. If you're on Windows with limited disk space, this is the setup I'd use:

- Use WSL2. It gives you a real Linux environment inside Windows (`wsl --install` in an administrator PowerShell, then restart). A large share of the tutorials and answers in this field assume Linux, and you'll stop fighting your environment.
- One environment per project, created with `uv` or `python -m venv`, so libraries from different projects don't collide. If disk space is tight, keep one shared environment for learning and create separate ones only for projects you'll keep.
- Install the CPU-only build of PyTorch. On Linux, and WSL2 is Linux, a plain `pip install torch` from PyPI brings the GPU (CUDA) libraries with it. When I installed it for this addendum, on a machine with no GPU, that was about 4.2 GB: 1.2 GB of PyTorch and 3.0 GB of NVIDIA libraries that will never run. The CPU-only build comes from PyTorch's own package index and leaves all of that out (the command is right after this list).
- Clean the caches now and then. `pip cache purge` (or `uv cache clean`) frees the downloaded packages, and Hugging Face models are cached in `~/.cache/huggingface` (you can move it with the `HF_HOME` environment variable). Prefer small models while learning: the one in [chapter G](G-sentence-transformers-in-miniature.md) is under 100 MB.
- Borrow a GPU when you need one. Free notebooks on Google Colab and Kaggle give you a GPU for a few hours at a time, which is enough for everything a learner needs, including fine-tuning small models.

The CPU-only install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The selector on PyTorch's "Get Started" page gives you the exact command for your system.

## 6. Using an assistant without losing the learning

[Section B.8](B-a-method-for-learning-any-library.md#8-typing-paper-and-assistants-a-protocol) gave the rule: tutor, not ghostwriter. In practice, these are prompts that keep the learning on your side:

- *"I'm learning scikit-learn's estimator API. Ask me five questions about it, one at a time, and don't tell me the answer until I've tried."*
- *"Here's my code and the error. Don't fix it. Tell me which line to look at and what concept I should check."*
- *"Explain what this function does with each of its arguments, and point me to the part of the documentation where that's described."*
- *"Here's my miniature of X. What behavior of the real library does it not reproduce? Give me an input where they'd disagree."* (Then check, because the assistant can be wrong about that too.)

And one habit for the code an assistant gives you when you're not learning but working: before you use it, predict what each line does. If you can't, that's the line to read about, because it's the one most likely to bite you later.

**Where this shows up in No Shortcuts:** lesson 00.2, "Your workstation: Linux, shell, Git, Python environments, Docker", is the full setup, and the four capstones in Part IX are bigger versions of the projects in [section H.4](#4-projects-that-force-you-to-use-them).

---

[← G. sentence-transformers in miniature](G-sentence-transformers-in-miniature.md) · [Contents](../README.md#contents) · [I. FAQ and cheat sheet →](I-faq-and-cheat-sheet.md)
