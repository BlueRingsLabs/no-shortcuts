# I. FAQ and cheat sheet

## 1. FAQ

These are the questions from the post that started all this, plus the ones I'd expect after reading it.

**Should I start with the docs, a tutorial, a video or a project?**

One short tutorial, to get something running and see the library's shape. Then the explanation pages ([section B.2](B-a-method-for-learning-any-library.md#2-documentation-comes-in-four-kinds-and-each-answers-a-different-question)) for the data model and the conventions. Then a miniature, and then a small project without a tutorial. Videos are fine for motivation and for seeing someone think, but they're the most passive of all of these, and the copy-paste phase feeds on passive.

**How do I read API documentation without drowning?**

Don't read it to learn; read it to look things up. When you do, read the signature, the shapes in the parameters and returns, and the Notes section, in that order ([section B.3](B-a-method-for-learning-any-library.md#3-how-to-read-a-reference-page)). Learn the library from the explanation pages and the user guide, which are short.

**How much should I memorize? Is it normal to look up syntax all the time?**

Completely normal, even after years. Memorize the data model and the conventions of each library you use (most of this addendum), and let the everyday functions stick by themselves through use. For the first kind, test yourself instead of re-reading ([section B.6](B-a-method-for-learning-any-library.md#6-how-much-should-you-memorize)).

**Which libraries should I learn, and in what order?**

NumPy, pandas, enough matplotlib to look at your data, scikit-learn, PyTorch, then Hugging Face and the RAG tools ([section H.3](H-from-copying-to-writing.md#3-which-libraries-in-what-order)). The order matters more than the speed.

**How do I go from copying lab code to writing my own?**

Type it instead of pasting it, predict before you run, rewrite it from memory the next day, translate it into another library, break it on purpose, and test what you believe about it ([section H.2](H-from-copying-to-writing.md#2-six-habits-that-make-the-jump)). Above all, learn the model under the library, because that's what lets you write code for situations no example covered.

**What projects force me to use these libraries for real?**

The six in [section H.4](H-from-copying-to-writing.md#4-projects-that-force-you-to-use-them), each with a "done when". All of them run on a laptop without a GPU.

**I'm on a normal Windows laptop with little storage. Is that enough?**

Yes, for everything in this addendum and most of what comes after. Use WSL2, install the CPU-only build of PyTorch, prefer small models, clean the caches, and borrow a free GPU from Colab or Kaggle when you need one ([section H.5](H-from-copying-to-writing.md#5-a-setup-for-a-normal-laptop)).

**Isn't using an AI assistant just the modern way of learning?**

It's the modern way of *working*, and you should get good at it. While you're learning a library, though, the evidence points the same way as common sense: when the assistant produces the code, you learn less (the introduction cites a randomized study on exactly this). Use it to quiz you and to explain, and keep the writing for yourself ([section H.6](H-from-copying-to-writing.md#6-using-an-assistant-without-losing-the-learning)).

**What if my miniature disagrees with the real library?**

That's a good outcome. Either you've misunderstood something (now you know exactly what), or you've found a corner of the library whose behavior surprised you, which is worth knowing before it surprises you in production. [Section F.8](F-pytorch-in-miniature.md#8-dataset-and-dataloader) has a real example: our data loader produced the same batches in a different order, and the reason turned out to be a detail of how PyTorch seeds its worker processes. [Section G.7](G-sentence-transformers-in-miniature.md#7-checking-the-miniature-and-a-disagreement-i-didnt-expect) has a bigger one: two implementations of "BERT's tokenizer" that disagree on a quarter of Unicode.

**The tutorial I'm following doesn't work with my versions. What do I do?**

Check the dates first: the tutorial's, and the release of each library you have installed. Then read the release notes in between, at least the sections on API changes and deprecations, and run your code with deprecation warnings turned on, because by default Python hides most of them ([section B.9](B-a-method-for-learning-any-library.md#9-how-a-library-tells-you-its-changing)). Most "this function doesn't exist" and "unexpected keyword argument" errors in old tutorials are exactly that.

**Why build miniatures instead of reading the real source code?**

Because a real library's source is mostly the part that isn't the idea: input checks, special cases, performance, compatibility, error messages. Reading it first, you can't tell the core from the rest. After building the miniature you can, and then reading the real source becomes easy and very instructive ([section B.4](B-a-method-for-learning-any-library.md#4-ask-the-library-about-itself)).

**How long does all of this take?**

Each chapter is a few hours if you type the code, run the tests and do some exercises; a weekend per library if you also do a project. That's slower than following a tutorial, and what you learn carries over to the next version of the library, and to the next library.

**Do I need the No Shortcuts course to use this?**

No. This addendum stands on its own. The course is where these libraries get used for real (models, training, evaluation, production), and the end of each chapter points to the relevant part.

## 2. Cheat sheet

Everything here was used somewhere in the addendum and checked on the versions listed in the introduction.

**Asking a library about itself**

| Goal | Code |
|---|---|
| Docstring | `help(obj)`, or `obj?` in Jupyter |
| Source | `inspect.getsource(obj)`, or `obj??` in Jupyter |
| Signature | `inspect.signature(f)` |
| Where it's defined | `inspect.getfile(obj)`, `obj.__module__` |
| What it inherits from | `cls.__mro__` |

**NumPy**

| Goal | Code |
|---|---|
| Inspect layout | `a.shape`, `a.dtype`, `a.strides`, `a.flags` |
| Do two arrays share memory? | `np.shares_memory(a, b)` |
| Reshape without copying, or fail | `a.reshape(shape, copy=False)` |
| Independent copy | `a.copy()` |
| Broadcast a reduction back | `a - a.mean(axis=1, keepdims=True)` |
| Broadcast without copying (read-only) | `np.broadcast_to(a, shape)` |
| A random generator to pass around | `rng = np.random.default_rng(seed)` |
| Independent streams | `rng.spawn(n)` |

**pandas**

| Goal | Code |
|---|---|
| Label vs position | `s.loc[label]`, `s.iloc[position]` |
| Check the index | `s.index`, `s.index.is_unique`, `a.index.equals(b.index)` |
| Add with missing labels as 0 | `a.add(b, fill_value=0)` |
| Assign to a subset (one step) | `df.loc[mask, "col"] = value` |
| Groups in order of appearance | `df.groupby("key", sort=False)` |
| One value per row, by group | `df.groupby("key")["col"].transform("mean")` |
| Check a join | `a.merge(b, on="k", validate="many_to_one")` |
| Fraction missing | `df.isna().mean()` |
| Read IDs and zip codes as text | `pd.read_csv(path, dtype={"zip": "str"})` |
| Parse dates strictly, then count failures | `pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").isna().sum()` |

**scikit-learn**

| Goal | Code |
|---|---|
| Check a custom estimator | `check_estimator(est, on_fail=None)` |
| Validate input in your estimator | `validate_data(self, X, y)` / `validate_data(self, X, reset=False)` |
| Is it fitted? | `check_is_fitted(self)` |
| Fresh unfitted copy | `clone(est)` |
| Preprocessing without leakage | `cross_val_score(make_pipeline(prep, model), X, y)` |
| See every tunable name | `sorted(pipe.get_params())` |
| Search a step's parameter | `GridSearchCV(pipe, {"ridge__alpha": [0.1, 1, 10]})` |
| Different preprocessing per column | `ColumnTransformer([("num", ..., num_cols), ("cat", ..., cat_cols)])` |
| Keep column names in the output | `prep.set_output(transform="pandas")` |
| Unseen categories at predict time | `OneHotEncoder(handle_unknown="ignore")` |

**PyTorch**

| Goal | Code |
|---|---|
| Inspect a tensor | `t.shape`, `t.dtype`, `t.device`, `t.stride()`, `t.is_contiguous()` |
| Reinterpret or fail / view-or-copy | `t.view(...)` / `t.reshape(...)` |
| Share memory with NumPy | `torch.from_numpy(a)` (copy: `torch.tensor(a)`) |
| How a tensor was made | `t.grad_fn`, `t.grad_fn.next_functions` |
| Layers in a loop | `nn.ModuleList([...])` |
| Count parameters | `sum(p.numel() for p in model.parameters())` |
| No tape during evaluation | `with torch.no_grad():` |
| Reproducible batch order | `DataLoader(..., generator=torch.Generator().manual_seed(0))` |
| Check a custom backward | `torch.autograd.gradcheck(f, inputs)` (with `float64`) |
| What's saved with the model | `model.state_dict().keys()` |
| Save and restore | `torch.save(model.state_dict(), path)` / `model.load_state_dict(torch.load(path))` |

**sentence-transformers**

| Goal | Code |
|---|---|
| What the model is made of | `print(model)` |
| How many tokens are read | `model.max_seq_length` |
| See the tokens | `model.tokenizer.convert_ids_to_tokens(model.tokenizer(text)["input_ids"])` |
| Cosine similarity of normalized vectors | `emb @ emb.T` or `model.similarity(emb, emb)` |

**Warnings**

| Goal | Code |
|---|---|
| Show deprecations from your own modules | `warnings.simplefilter("default", DeprecationWarning)` |
| Make deprecations fail | `python -W error::DeprecationWarning script.py` |

---

[← H. From copying to writing](H-from-copying-to-writing.md) · [Contents](../README.md#contents) · [J. Final thoughts →](J-final-thoughts.md)
