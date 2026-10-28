# B. A method for learning any library

Before the miniatures, the method. This chapter is the one you can apply to any library, including the ones that don't exist yet, and it answers the first questions in that Reddit post: where do I start, how do I read the docs, how much do I have to memorize.

## 1. A library has layers, and you learn them in a different order than you'd think

Every serious library can be read at five levels:

1. **The problem it solves.** NumPy exists because Python loops over numbers are slow and Python lists don't know what a matrix is. scikit-learn exists because every classical ML algorithm needs the same plumbing around it (input checking, train/test handling, hyperparameters, evaluation) and nobody wants to rewrite that for each one.
2. **The data model.** The one or two objects everything else operates on: NumPy's array, pandas' Series and DataFrame, PyTorch's tensor and module, scikit-learn's estimator.
3. **The conventions.** The rules that hold across the whole library: what gets modified in place and what returns a new object, how functions are named, what they return, how errors are reported. In scikit-learn, "learned things end with an underscore" is a convention; in pandas, "operations align on the index" is one; in NumPy, "slicing gives a view" is one.
4. **The API.** The thousands of individual functions and classes.
5. **The implementation.** The source code.

Tutorials start at level 4, accumulating functions, and that's usually where learners stop too. I think that's backwards. Levels 1 to 3 are small (you can learn them in an afternoon or two per library), and once you have them, level 4 stops being something to memorize. When you meet a function you've never seen, you already know what kind of object it takes, whether it'll modify your data, and what shape its result will have, because those things are decided by the conventions, not by the function. You only need level 5 when the library surprises you, and then it's the best teacher you'll find.

So the order is 1, 2, 3, then 4 as you need it, and 5 when something doesn't do what you expected. The miniatures in this addendum are levels 2 and 3 written as code.

## 2. Documentation comes in four kinds, and each answers a different question

A big reason documentation feels overwhelming is reading the wrong kind for what you need. A useful map is **Diátaxis**, a framework for technical documentation by Daniele Procida ([diataxis.fr](https://diataxis.fr/)<sup>\[6\]</sup>), which splits documentation into four kinds:

| Kind | It answers | Read it when |
|---|---|---|
| **Tutorial** | "Show me, step by step, so I get something working." | You've never used the library. Once. |
| **How-to guide** | "How do I do this specific thing?" | You have a task and know roughly what you need. |
| **Reference** | "What exactly does this function take and return?" | You're writing code and need a precise detail. |
| **Explanation** | "Why does it work this way? What's the idea?" | You want to understand the library, not just use it. |

Not every project labels its docs this way, but the four kinds are always there under other names. In NumPy, the user guide pages like "Copies and views" and "Broadcasting" are explanation, and the API reference is reference. In scikit-learn, the User Guide is explanation (it describes each family of models, with the math), the examples gallery is a big collection of how-to guides, and the API reference is reference. PyTorch has tutorials at one address and the API docs at another.

The mistake that makes docs feel impossible is going straight to the reference to *learn*. The reference is written for people who already understand the library and need one detail. Reading it to learn is like learning a language from its dictionary. The explanation pages are where the data model and the conventions live, they're usually short, and they're the easiest ones to skip.

> [!TIP]
> For every new library, find the explanation pages first and read them once, slowly, before you touch the API reference. For NumPy that's "the absolute basics", "copies and views" and "broadcasting". For scikit-learn it's the "Developing scikit-learn estimators" page, which, despite the name, is the clearest description of the estimator contract that exists. For pandas it's "Intro to data structures" and the indexing pages.

## 3. How to read a reference page

When you do go to the reference, there's a structure to it, and most libraries in the scientific Python world use the same one (the NumPy docstring format). It starts with the signature, and the signature is worth reading carefully, because it encodes rules. Here's the real one from NumPy 2.5:

```python
import inspect
import numpy as np

print(inspect.signature(np.reshape))
```

```text
(a, /, shape, order='C', *, copy=None)
```

The `/` means everything before it is **positional-only**: you can't write `np.reshape(a=x, ...)`. The `*` means everything after it is **keyword-only**: you must write `copy=False`, never pass it by position. Defaults tell you the library's opinion about the common case.

After the signature come the sections, from top to bottom:

- **Parameters.** Each one with its type and meaning. Types like `array_like` mean "anything NumPy can turn into an array" (a list, a tuple, another array), and `{array-like, sparse matrix} of shape (n_samples, n_features)` in scikit-learn tells you the expected **shape**, which is the single most useful piece of information on the page.
- **Returns.** What comes back, and its shape. Also check whether it's a new object or the same one modified.
- **Notes.** Often the most important section, and the one people skip: it's where the docs tell you about copies, numerical caveats and edge cases.
- **See also.** The neighbors of this function. When you're looking for something, the right function is very often in the "See also" of the wrong one.
- **Version notes.** Lines like `.. versionadded:: 0.24` or `.. versionchanged:: 2.1`. When code from a tutorial doesn't work on your machine, this is the first place to look, because tutorials age much faster than libraries.
- **Examples.** Small, runnable, and usually chosen to show the edge cases. Run them and change them.

You don't read a reference page from top to bottom. You read the signature, the shapes in the parameters and returns, and the notes. Everything else when you need it.

## 4. Ask the library about itself

Python lets you interrogate any object, and that's often faster and more precise than searching the web. These are the tools worth knowing:

| Tool | What it tells you |
|---|---|
| `help(obj)` or `obj?` in Jupyter | The docstring, which is the reference page, offline |
| `obj??` in Jupyter | The source code |
| `type(obj)` | What kind of object you actually have (not what you think you have) |
| `dir(obj)` | Every attribute and method |
| `inspect.signature(f)` | The exact signature |
| `inspect.getsource(f)` | The source of a function or class |
| `inspect.getfile(obj)`, `obj.__module__` | Where it's defined |
| `cls.__mro__` | The classes it inherits from, in order |

Let's interrogate something you've probably used without thinking much about it, scikit-learn's `StandardScaler`:

```python
import inspect
from sklearn.preprocessing import StandardScaler

print(inspect.signature(StandardScaler))
print(StandardScaler.__module__)
print([c.__name__ for c in StandardScaler.__mro__])
```

```text
(*, copy=True, with_mean=True, with_std=True)
sklearn.preprocessing._data
['StandardScaler', 'CallbackSupportMixin', 'OneToOneFeatureMixin', 'TransformerMixin', '_SetOutputMixin', 'BaseEstimator', 'ReprHTMLMixin', '_HTMLDocumentationLinkMixin', '_MetadataRequester', 'object']
```

Three lines, and you've learned a lot about scikit-learn's design:

- The constructor takes only keyword arguments (the `*` at the start), and only **hyperparameters**: no data. Data goes to `fit`, never to the constructor. That's a rule for every scikit-learn estimator, and [chapter E](E-scikit-learn-in-miniature.md) explains why.
- It lives in a private module (`_data`, with a leading underscore) and is re-exported from the public one. You import it from `sklearn.preprocessing`, and the underscore tells you the internal layout may change.
- It inherits from `TransformerMixin` (which is where `fit_transform` comes from, as you'll see) and from `BaseEstimator` (which gives it `get_params`, `set_params` and its printed representation). Most of what a scaler "can do" isn't written in the scaler at all; it comes from those mixins.

Now the source of `fit`:

```python
print(inspect.getsource(StandardScaler.fit))
```

The output is 26 lines, and most of them are the docstring. Without it, the method is this:

```python
    def fit(self, X, y=None, sample_weight=None):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)
```

So `fit` is "forget everything, then do one incremental update on all the data", and the real work is in `partial_fit`. That one look tells you three things that are easy to miss in a tutorial: `fit` can be called again and starts from scratch, there's a `partial_fit` for data that arrives in batches, and `fit` returns `self`, which is why `StandardScaler().fit(X).transform(X)` works as a chain.

Reading source is less scary than it sounds, for one reason: you almost never read a whole library. You read one function, the one that surprised you, and maybe the one it calls.

## 5. Follow the traceback into the library

The second great teacher is the error message, and it's tempting to read only its last line. Here's a very common first-week mistake with scikit-learn:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([1.0, 2.0, 3.0, 4.0])
score = np.array([52.0, 61.0, 68.0, 80.0])

model = LinearRegression()
model.fit(hours, score)
```

```text
Traceback (most recent call last):
  File "ch1_traceback.py", line 8, in <module>
    model.fit(hours, score)
  File ".../site-packages/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File ".../site-packages/sklearn/linear_model/_base.py", line 681, in fit
    X, y = validate_data(
  File ".../site-packages/sklearn/utils/validation.py", line 3058, in validate_data
    X, y = check_X_y(X, y, **check_params)
  File ".../site-packages/sklearn/utils/validation.py", line 1330, in check_X_y
    X = check_array(
  File ".../site-packages/sklearn/utils/validation.py", line 1076, in check_array
    raise ValueError(msg)
ValueError: Expected 2D array, got 1D array instead:
array=[1. 2. 3. 4.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```

(I've shortened the paths and removed the lines Python 3.13 adds under each frame to point at the failing expression.)

Read it from the bottom up:

1. **The message.** It's unusually helpful: scikit-learn wanted a 2D array and got a 1D one, and it even suggests the fix.
2. **The frame that raised.** `check_array` in `sklearn/utils/validation.py`. That's a function name worth remembering, because every scikit-learn estimator runs its input through it. If you open it (`inspect.getsource`), you'll find all the checks scikit-learn applies to your data: dimensions, finite values, dtypes, sparse formats.
3. **The path from your code to there.** `fit` called `validate_data`, which called `check_X_y`, which called `check_array`. So the check happens at the very start of `fit`, before any learning.
4. **The last frame in your code.** `line 8, model.fit(hours, score)`. That's the line you change.

scikit-learn insists on 2D because the convention is **rows are samples, columns are features**, always, even when there's one feature. A 1D array is ambiguous: four samples with one feature, or one sample with four features? scikit-learn refuses to guess, on purpose. The fix:

```python
model = LinearRegression().fit(hours.reshape(-1, 1), score)
print(model.coef_, model.intercept_, model.n_features_in_)
print(model.predict(np.array([[5.0]])))
```

```text
[9.1] 42.5 1
[88.]
```

`coef_`, `intercept_` and `n_features_in_` end with an underscore because they were *learned* by `fit`. That's the convention from [section B.1](#1-a-library-has-layers-and-you-learn-them-in-a-different-order-than-youd-think) again, and you'll implement it yourself in chapter E.

> [!TIP]
> When a traceback goes into library code, find two frames: the **last one in your code** (that's what you change) and the **one that raised** (that's what you read to understand why). Everything in between is usually plumbing.

## 6. How much should you memorize?

Much less than you think, but not nothing. Looking up syntax all the time is completely normal, even after years of using a library. What experienced people don't look up is a different kind of knowledge:

- The data model and the conventions (levels 2 and 3). These you should know cold, because they're what lets you predict what an unfamiliar function will do. This addendum is mostly about these.
- The twenty or thirty functions you use every day. You'll memorize these without trying, by using them. Don't make flashcards of them.
- Where to look for everything else. Knowing that NumPy's reductions all take `axis` and `keepdims`, or that scikit-learn's metrics live in `sklearn.metrics`, is worth more than remembering any individual signature.

For the first kind, active recall works and re-reading doesn't. Learning research is unusually clear about this: testing yourself (trying to remember without looking) beats reading the same material again, and spacing that practice over days beats cramming it into one evening ([Dunlosky et al., 2013](https://doi.org/10.1177/1529100612453266)<sup>\[7\]</sup> review the evidence for ten common study techniques; practice testing and distributed practice are the two they rate highest). In practice:

- Write questions that make you *produce* something, not recognize it. Not "what is broadcasting?" but "what shape does `(5, 1)` plus `(3,)` give, and why?". Not "what does `fit` return?" but "why can I write `StandardScaler().fit(X).transform(X)`?".
- Answer them on paper, from memory, a few days later. Check. The ones you got wrong are the ones to keep.
- Keep a cheat sheet that *you* write, for the functions you keep looking up. Writing it is where most of the learning happens (the generation effect again), so a downloaded one does much less for you.

## 7. The rebuild-and-check loop

This is the method the rest of the addendum follows, written as steps you can apply to any library:

1. **Find the core.** Read the explanation pages and the type of the main object until you can say, in one or two sentences, what the data model is and what the main conventions are. If you can't say it, you haven't found it yet.
2. **Write a miniature from memory.** Twenty to a hundred lines of plain Python that implement that core and nothing else. Close the docs while you write it. Getting stuck is information: it tells you exactly what you didn't understand.
3. **Write a test against the real library.** Not three examples you chose; many random inputs, compared with what the library does ([section C.12](C-numpy-in-miniature.md#12-checking-the-miniature-against-the-real-thing) shows how). Use the library's own testing tools when it has them: `np.testing`, `pandas.testing`, scikit-learn's `check_estimator`, `torch.testing` and `torch.autograd.gradcheck`.
4. **Run it and read the failures.** Each failure is a precise statement of something you believed about the library that isn't true.
5. **Read the real source of whatever you got wrong.** Now you know what you're looking for, which makes reading library code ten times easier.
6. **Use the real library on a small project**, without a tutorial. You'll notice that you write code you've never seen, because you're no longer recalling examples; you're applying the model.

It takes longer than following a tutorial, a few hours per library instead of one, and what you get for those hours is a model that keeps working when the situation stops looking like the tutorial's.

## 8. Typing, paper and assistants: a protocol

Going back to the point from the introduction, here's how I'd use all of this while learning a library:

- Type everything, never paste. Including the code in this addendum. Typing makes you read every character, and the typos you make teach you what the error messages mean.
- Change one thing and predict before you run. A different shape, a different argument, a wrong type. Write down what you expect, then run it. Being wrong about a prediction is great news while you're learning: it's a gap between your model of the library and the library, found on a Tuesday afternoon instead of in production.
- The next day, write the miniature again on paper, from memory. Then type it and run the test. Whatever you couldn't remember is what you go back to.
- Use assistants as a tutor, not as a ghostwriter. "Explain why this broadcast fails" is fine. "Ask me five questions about scikit-learn's estimator contract and don't give me the answers until I've tried" is excellent, and it's retrieval practice with infinite questions. "Write the training loop for me" is skipping exactly the part that builds the skill. And check what it tells you against the docs or the source, because assistants are confidently wrong more often than you'd like, especially about the details that changed between versions.

> [!NOTE]
> None of this means you should never use an assistant to write code. At work you'll probably use one a lot, and that's fine. The point is narrower: while you're learning a library, the effort of producing the code is the learning, so handing that effort to a machine hands the learning to it too.

## 9. How a library tells you it's changing

Everything in this addendum is true for specific versions, and some of it wasn't true two years ago. pandas 3.0 made copy-on-write the only behavior and gave text its own `str` dtype ([chapter D](D-pandas-in-miniature.md)), NumPy 2 changed how Python numbers mix with arrays ([chapter C](C-numpy-in-miniature.md)), and the PyTorch DataLoader detail in [chapter F](F-pytorch-in-miniature.md) could change in any release. Plenty of tutorials, Stack Overflow answers and assistant suggestions were written against older versions and never updated. So part of learning a library is learning how it announces changes, and listening.

Libraries announce them in two places: the **release notes** (pandas calls them "What's new", scikit-learn "Release History", NumPy "Release notes", and PyTorch publishes them on its GitHub releases page), and **warnings** in your own output, months before the change. The big libraries have written policies for this, short enough to read:

- NumPy (NEP 23): a deprecated feature is removed "after at least 2 releases", which with two releases a year is about a year. Behavior that will *change*, rather than disappear, gets a `FutureWarning`.
- scikit-learn: a renamed class, function or parameter keeps working "for two releases" with a warning.
- pandas (its "Policies" page and PDEP-17): deprecations are only enforced in major versions. Something deprecated in 3.1 works, with a warning, for the whole 3.x series, and changes in 4.0.

That last policy comes with a detail that matters a lot in practice. Here's a warning from pandas 3.1:

```python
import pandas as pd

idx = pd.date_range("2026-01-01", periods=5, freq="D")
print(repr(pd.infer_freq(idx)))
```

```text
freq_demo.py:4: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
  print(repr(pd.infer_freq(idx)))
'D'
```

A good warning: it says what will change, when (pandas 4), how to get the new behavior today, and how to write code that works with both. `Pandas4Warning` is a subclass of Python's `DeprecationWarning`, and **Python hides `DeprecationWarning` unless the code that triggers it is in `__main__`**, the script you ran or the notebook cell you typed. The pandas policy page says this itself: new deprecations start hidden for code inside imported modules, so the libraries built on pandas get time to adapt before end users are told. In the final minor release before 4.0 they become `FutureWarning`s, which Python always shows.

You can see the hiding with NumPy 2.5, which deprecated assigning to `a.shape`. Put the assignment in your own module:

```python
# lib_mod.py
def reshape_in_place(a):
    a.shape = (2, 3)
    return a
```

Run it from a script, through `lib_mod`, and nothing appears. Write `a.shape = (2, 3)` directly in the script and the warning appears:

```text
DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
As an alternative, you can create a new view using np.reshape (with copy=False if needed).
```

So the deprecations in your project's modules, the code you actually maintain, are the ones you don't see. Three ways to see them:

- pytest shows them by default. The same `lib_mod` under a one-line test gives "1 passed, 1 warning", with the full message in the warnings summary. One more reason to have tests ([section B.7](#7-the-rebuild-and-check-loop)).
- `python -W error::DeprecationWarning script.py` turns them into exceptions, with a traceback pointing at the exact line. It's the strict version, good for a CI job. (`-W error` alone turns *every* warning into an error. Importing NumPy, pandas, scikit-learn, PyTorch and sentence-transformers at the versions in this addendum with `-W error` is clean, so you can afford it.)
- `warnings.simplefilter("default", DeprecationWarning)` at the top of a script shows each one once per location, including the ones from your modules, without stopping anything.

And when you upgrade a library, read the release notes of every version you skipped, at least the sections titled something like "API changes", "Deprecations" and "Removals". It's usually a few minutes of reading, and much cheaper than finding out from a wrong result. [Section D.6](D-pandas-in-miniature.md#6-copy-on-write-the-change-that-broke-old-tutorials) is about one change, copy-on-write, that changed what a lot of old pandas code does.

> [!TIP]
> When code from a tutorial or an assistant doesn't work, check its age before you debug it. "The function doesn't exist", "unexpected keyword argument" and a result that's wrong for no visible reason are, very often, version problems. `pip show numpy` (or `np.__version__`) for your side, the date of the post for theirs, and the release notes in between.

**Where this shows up in No Shortcuts:** lesson 00.3, "How to study this without lying to yourself", is the full version of [sections B.6](#6-how-much-should-you-memorize) to [B.8](#8-typing-paper-and-assistants-a-protocol), with the evidence and a protocol for using assistants during the course.

---

[← A. Introduction](A-introduction.md) · [Contents](../README.md#contents) · [C. NumPy in miniature →](C-numpy-in-miniature.md)
