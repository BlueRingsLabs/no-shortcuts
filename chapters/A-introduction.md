# A. Introduction

## 1. Abstract

Some people learning machine learning reach a stage where they can run code copied from tutorials but can't write it themselves. This addendum argues that one cause is the way libraries are usually learned: function by function, from the outside, without the small data model and design conventions each library is built on. For NumPy, pandas, scikit-learn, PyTorch and sentence-transformers, we identify that core, rebuild it as a miniature in plain Python, and check the miniature against the real library, using the library's own test tools where they exist (for example, scikit-learn's `check_estimator`). Each chapter ends with the documentation worth reading, exercises, and the part of the No Shortcuts course that builds on it. The last chapters turn this into a study method: how to read documentation, what to memorize, and how to move from copying code to writing it. All code was run, and every output shown is real.

## 2. The question

Someone on r/learnmachinelearning asked a question a few days ago that I suspect more people have than say out loud. Paraphrasing: *"I copy the code from the labs and tutorials and it works, but I can't write it myself. When I see `fit()` and `predict()` or `model.encode(...)` I understand what they do in that example, and I'm lost as soon as the situation is a little different."*

I know that feeling very well. My first real ML job had very little to do with the ML I had studied. They gave me a table with 400 million rows and half the labels broken, the pipeline blew up every five minutes, and the courses I had paid for (three of them, my wallet still remembers xD) were useless for all of it. They had taught me which function to call. None of them had taught me what the function was doing, so the moment the data stopped looking like the tutorial's, I was copying code I didn't understand into a problem it wasn't written for.

This addendum is my answer to that question, and it's long on purpose (sorry, not sorry :B).

## 3. Where the copy-paste phase comes from

A common way to learn a library, and the way I learned my first ones, is from the outside, one function at a time. You see `df.groupby("city").mean()` in a tutorial, it works, you remember it. You see `StandardScaler().fit_transform(X)`, it works, you remember that too. After a few months you have a long list of incantations and no model of what's underneath, so every new situation looks like a new incantation you haven't seen yet.

The thing is that these libraries are not long lists. Each one is a **small set of ideas**, a data model plus a handful of design conventions, with thousands of functions built on top:

- **NumPy** is one idea: an array is a block of memory plus four numbers that say how to read it. Slicing, transposing, broadcasting, reshaping and "why did my original array change?" all come out of that one idea.
- **pandas** adds one more: every column carries an index, and operations line up by label, not by position. Most pandas surprises are that alignment doing its job.
- **scikit-learn** is a contract. Hyperparameters go in the constructor, `fit` learns from data and stores the result in attributes that end with an underscore, `predict` and `transform` use what was learned. The whole library, pipelines and cross-validation included, is built on that contract.
- **PyTorch** is tensors (NumPy's idea again) plus a recorder that writes down every operation so it can compute gradients backwards, plus `nn.Module`, which is mostly a clever way of collecting parameters.
- **sentence-transformers** is a tokenizer, a transformer, an average and a normalization. `model.encode()` is about fifteen lines once you know that.

When you know the idea, the documentation changes. It stops being a wall of functions and becomes a list of things you can already guess the behavior of. That's the jump from copying code to writing it.

## 4. And now the code comes from an AI

There's a second cause, and it's newer. A few years ago the code you copied came from a tutorial or a Stack Overflow answer, and at least you had to find it, read it a bit, and adapt it. Now you ask an assistant, it hands you thirty lines that run, you paste them, and nobody ever types a single character of them. The code works, so it feels like progress, and if what you wanted was to learn the library, it mostly isn't.

This one has been measured. In a randomized experiment published in January 2026, developers learned a new Python library (Trio, for asynchronous code), some with an AI assistant and some without ([Shen and Tamkin, "How AI Impacts Skill Formation"](https://arxiv.org/abs/2601.20245)<sup>\[1\]</sup>; both authors work at an AI company, which is worth knowing when you read it). The group with AI understood the library worse, read and debugged code worse, and wasn't significantly faster on average. Those who handed the whole task to the assistant learned the least. The ones who kept their heads in it (asking for explanations, checking what they got) kept most of the learning. It's one study with 52 people and one library, so don't treat it as a law of nature, but it points the same way as decades of memory research.

That research has a name for the effect: the **generation effect**. You remember what you produce much better than what you only read. It was described in 1978 ([Slamecka and Graf](https://doi.org/10.1037/0278-7393.4.6.592)<sup>\[2\]</sup>), and a meta-analysis of 86 studies later found the advantage again and again, close to half a standard deviation on average ([Bertsch et al., 2007](https://doi.org/10.3758/BF03193441)<sup>\[3\]</sup>). Pasting code is reading it, at best. Typing it is a little better. Writing it from memory, and only then checking, is generating it.

Nobody can stop you from pasting, and at work you'll probably use assistants a lot, which is fine. But while you're *learning* a library, type every character yourself. I'd go further: write some of it with pencil and paper. I know it sounds old-fashioned. To be honest about the evidence, the famous "the pen beats the laptop" study about note-taking didn't hold up well when others repeated it ([Morehead, Dunlosky and Rawson, 2019](https://doi.org/10.1007/s10648-019-09468-2)<sup>\[4\]</sup>), so paper isn't magic. What paper does is take away autocomplete, copy-paste and the assistant, so the only way to get the code onto the page is to generate it, which is the part the evidence supports. Every chapter in this addendum is built so you can do exactly that: small enough to write from memory, and with a test that tells you whether you got it right.

## 5. The method: build it small, then check it against the real thing

The way I know to get those ideas into your head (not just read about them) is to **rebuild each library in miniature**: a few dozen lines of plain Python that implement the data model and the core conventions. Then, and this is the important part, **check your miniature against the real library**, with the library's own tools when it has them:

- the NumPy miniature is checked against NumPy on hundreds of random shapes, slices and broadcasts;
- the scikit-learn estimators we write are run through `check_estimator`, the same test suite scikit-learn uses on its own estimators;
- the PyTorch autograd is checked against PyTorch's gradients;
- the hand-written `encode()` is compared number by number with what sentence-transformers returns.

If your miniature passes, you understand the core of the library, the part every function you'll ever call depends on. If it fails, the failure tells you exactly which part you had wrong, and that's worth more than a pass.

The miniatures are slow and incomplete on purpose. They're not meant to replace anything. They're meant to be small enough that you can hold the whole thing in your head.

> [!NOTE]
> Every snippet in this addendum was run before publishing, and every output shown is the real output. The versions used: Python 3.13.17, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, PyTorch 2.15.0 (on CPU), sentence-transformers 6.2.0 with transformers 5.21.0. If you use older versions, a few details change (the text points them out where it matters).

## 6. Who this is for

You know Python: functions, classes, lists, dictionaries, a `for` loop that doesn't scare you. You've used at least one of these libraries a bit, probably by copying code that worked. You don't need a GPU, and you don't need a powerful laptop; everything here runs on a CPU in seconds, and [chapter H](H-from-copying-to-writing.md) has a setup for a modest Windows machine.

You don't need to read it in order, either. Every chapter stands on its own, so if `fit_transform` is what bothers you, jump straight to scikit-learn.

## 7. What's in it

- **B. A method for learning any library**: how a library is layered, how documentation is organized and which part to read for what, the introspection tools that let you ask a library about itself, how to follow a traceback into library code, what to memorize (much less than you think), and how a library warns you before it changes.
- **C. NumPy in miniature**: buffer, dtype, shape and strides; views and copies; broadcasting as a stride of zero; `einsum` as named loops; dtypes and overflow; memory layout and speed; random generators.
- **D. pandas in miniature**: the index, alignment, copy-on-write, split-apply-combine, joins that multiply rows, why `apply` is slow, and what `read_csv` guesses about your data.
- **E. scikit-learn in miniature**: the estimator contract, our own estimators passing `check_estimator`, pipelines and why `fit_transform` on the test set leaks, a grid search in twenty lines, and real tables with `ColumnTransformer`.
- **F. PyTorch in miniature**: tensors, autograd as a tape, how `nn.Module` finds your parameters, optimizers, the training loop, the four things that break real loops (dtypes, in-place operations, `train()`/`eval()`, `state_dict`), and `DataLoader`.
- **G. sentence-transformers in miniature**: a WordPiece tokenizer, BERT in about forty lines of NumPy with the real weights, pooling and its padding bug, the 256-token wall, and two tokenizers that disagree on a quarter of Unicode.
- **H. From copying to writing**: exercises, the order to learn things in, project ideas, and a setup for a normal laptop.
- **I. FAQ and cheat sheet**, then a few final words (**J**) and the references (**K**).

## 8. Why this is a No Shortcuts addendum

This is the first addendum to **No Shortcuts**, a free course on data, ML, deep learning and AI engineering that I've been writing ([github.com/BlueRingsLabs/no-shortcuts](https://github.com/BlueRingsLabs/no-shortcuts)<sup>\[5\]</sup>). You're reading it in its own branch of the course repository, and every miniature, with the tests that check it against the real library, is in [`code/`](../code/). That's the "companion code" the chapters refer to. It isn't part of the course, and you don't need the course to read it. It covers something the course takes for granted.

The course rebuilds the *algorithms* from scratch before using the libraries: linear models before scikit-learn, an autograd engine before PyTorch, a GPT before Hugging Face. What it never does is open the libraries themselves and look at how they're designed, and that's the part people get stuck on when they copy code. This addendum does the same thing the course does, with the *tools* instead of the algorithms. At the end of each [chapter I](I-faq-and-cheat-sheet.md) point to the part of the course where that library does the heavy lifting, in case you want to keep going.

Sooo, let's go :B

---

[Contents](../README.md#contents) · [B. A method for learning any library →](B-a-method-for-learning-any-library.md)
