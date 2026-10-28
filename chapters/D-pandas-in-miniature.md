# D. pandas in miniature

If NumPy is an array and four numbers, pandas is NumPy plus **labels**. Every Series carries an index, every DataFrame carries an index for its rows and one for its columns, and almost every operation lines up data **by label** before it does anything else. That one idea explains most of the things that make pandas feel unpredictable, including the NaNs that appear out of nowhere and the assignments that don't assign.

pandas 3.0 also changed the rules about copies, so plenty of older tutorials, and plenty of assistant-generated code, still show behavior that no longer exists. This chapter uses pandas 3.1.0 and points out where older versions behaved differently.

## 1. A Series is values plus an index

Here are two Series with partly overlapping labels:

```python
import pandas as pd

s1 = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
s2 = pd.Series([10.0, 20.0, 30.0], index=["b", "c", "d"])
print(s1 + s2)
```

```text
a     NaN
b    12.0
c    23.0
d     NaN
dtype: float64
```

pandas didn't add the first element to the first element. It added `"b"` to `"b"` and `"c"` to `"c"`, and for `"a"` and `"d"`, which only exist on one side, it had nothing to add, so it wrote NaN. The result has the union of the two indexes.

Now drop the labels and add the raw values:

```python
print(s1.values + s2.values)
```

```text
[11. 22. 33.]
```

Same data, a completely different answer, and no warning. This is the pandas version of the square-matrix bug from [section C.7](C-numpy-in-miniature.md#7-a-broadcasting-bug-that-doesnt-crash): going down to `.values` (or `.to_numpy()`) throws the labels away, and NumPy adds by position. If the two Series were sorted differently, or one was missing a row, the result is silently wrong.

> [!WARNING]
> `.values` and `.to_numpy()` drop the index. Do arithmetic on Series or DataFrames and let pandas align, or make sure the indexes are identical first (`s1.index.equals(s2.index)`) before going down to NumPy.

If you want missing labels treated as zero instead of NaN, the method form of the operator takes a `fill_value`:

```python
print(s1.add(s2, fill_value=0))
```

```text
a     1.0
b    12.0
c    23.0
d    30.0
dtype: float64
```

## 2. Building MiniSeries

The miniature keeps the values in a list and the labels in another list. Alignment is a dictionary lookup per label:

```python
NAN = float("nan")


class MiniSeries:
    def __init__(self, values, index=None, name=None):
        self.values = list(values)
        self.index = list(index) if index is not None else list(range(len(self.values)))
        self.name = name

    def _aligned(self, other):
        if self.index == other.index:            # same labels in the same order: keep them
            return self.index
        labels = set(self.index) | set(other.index)
        try:
            return sorted(labels)                # otherwise the sorted union, like pandas
        except TypeError:                        # labels that can't be compared
            return list(dict.fromkeys(self.index + other.index))

    def _binary(self, other, op, fill_value=None):
        index = self._aligned(other)
        mine = dict(zip(self.index, self.values))
        theirs = dict(zip(other.index, other.values))
        out = []
        for label in index:
            a, b = mine.get(label, fill_value), theirs.get(label, fill_value)
            out.append(NAN if a is None or b is None else op(a, b))
        return MiniSeries(out, index)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def add(self, other, fill_value=None):
        return self._binary(other, lambda a, b: a + b, fill_value)
```

(The full version in the companion code keeps the values in a small `Buffer` object, for the copy-on-write section below. The logic of alignment is exactly this.)

```python
s1 = MiniSeries([1.0, 2.0, 3.0], ["a", "b", "c"])
s2 = MiniSeries([10.0, 20.0, 30.0], ["b", "c", "d"])
print((s1 + s2).index, (s1 + s2).values)
print(s1.add(s2, fill_value=0.0).values)
```

```text
['a', 'b', 'c', 'd'] [nan, 12.0, 23.0, nan]
[1.0, 12.0, 23.0, 30.0]
```

Same as pandas.

## 3. Alignment is a join

It helps to see alignment for what it is: every binary operation between two Series is an **outer join** on the index, followed by the operation on the joined values. That's why the result has the union of the labels, and why the labels that only appear on one side get NaN, which is what an outer join gives you for the side that has no match.

It's a surprising default, and a deliberate one: pandas was built for data that comes from different places at different times. Two price series from two exchanges, a table of sales and a table of returns, a column computed on a filtered subset and added back to the full table: in all of those, adding by position would be a bug, and adding by label is what you meant. The NaNs aren't pandas being difficult. They're pandas telling you that some labels didn't match, which is information you want.

When an operation doesn't do what you expected, the first thing to check is almost always the index: `s.index`, whether it has duplicates (`s.index.is_unique`), whether two indexes are equal (`a.index.equals(b.index)`), and whether a previous filter left it with gaps.

## 4. Labels and positions are different things

The index is labels, not positions, and pandas gives you two separate tools to say which one you mean: `.loc` for labels and `.iloc` for positions. Plain brackets `s[...]` on a Series mean **labels**. With string labels it's hard to get confused. With integer labels it's easy:

```python
s = pd.Series([5, 6, 7], index=[2, 1, 0])
print(s[0], s.iloc[0])
```

```text
7 5
```

`s[0]` is the element whose **label** is 0 (the last one, 7). `s.iloc[0]` is the element at **position** 0 (the first one, 5). An integer index that isn't `0, 1, 2, ...` in order is exactly what you get after sorting or filtering a DataFrame, so this bug tends to appear in code that worked fine until someone added a `sort_values()` upstream.

The miniature makes the difference explicit, because it has to:

```python
    def _position(self, label):
        try:
            return self.index.index(label)
        except ValueError:
            raise KeyError(label) from None

    def loc(self, label):
        return self.values[self._position(label)]

    def iloc(self, i):
        return self.values[i]

    def __getitem__(self, label):               # like pandas: [] means label
        return self.loc(label)
```

> [!TIP]
> Write `.loc[...]` or `.iloc[...]` every time, even when plain brackets would work. It costs four characters and states your intent, which is exactly what you'll want when you read the code in three months.

## 5. A DataFrame is a dict of Series that share an index

A DataFrame is closer to a dictionary than to a matrix: a set of named columns, each one a Series, all sharing the same row index. Each column has its own dtype, which is why a DataFrame can hold numbers next to strings next to dates, something a NumPy array can't do.

```python
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
print(df.dtypes)
```

```text
city      str
temp    int64
dtype: object
```

That `str` is new. Since pandas 3.0, text columns get a dedicated string dtype by default, where older versions stored them as generic Python objects (`object`). If code from a tutorial checks `df["city"].dtype == object`, that's why it breaks now.

The miniature DataFrame is literally a dict of columns:

```python
class MiniDataFrame:
    def __init__(self, columns, index=None):
        n = len(next(iter(columns.values())))
        self.index = list(index) if index is not None else list(range(n))
        self._cols = {name: MiniSeries(vals, self.index, name) for name, vals in columns.items()}

    def __getitem__(self, name):
        return self._cols[name]._share()        # section D.6 explains _share
```

## 6. Copy-on-write: the change that broke old tutorials

This is the most important section of the chapter if you learned pandas before 2025, or from material written before then.

The question is simple: when you select part of a DataFrame and change it, does the original change? In NumPy ([section C.3](C-numpy-in-miniature.md#3-slicing-is-arithmetic-on-strides)), a slice is a view, so yes. In pandas up to version 2, the answer was "it depends": some selections returned views and some returned copies, the rules were hard to predict, and pandas warned you with the famous `SettingWithCopyWarning` because often even pandas couldn't be sure.

pandas 3.0 replaced all of that with one rule, **copy-on-write**: every selection behaves like a copy. The original never changes unless you modify it directly. In pandas 3.1, copy-on-write can't even be turned off; the old option to control it only produces a deprecation warning.

Look at what that does to the classic pattern:

```python
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
df[df["temp"] > 10]["temp"] = 0
print(df)
```

```text
ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment.
Such chained assignment never works to update the original DataFrame or Series, because the intermediate object on which we are setting values always behaves as a copy (due to Copy-on-Write).

Try using '.loc[row_indexer, col_indexer] = value' instead, to perform the assignment in a single step.
   city  temp
0  Rome    20
1  Lima    18
2  Rome    24
3  Oslo     5
```

(Those first lines are a warning, not an exception, so the code keeps running, and `df` is unchanged. I've left out the last line of the warning, a link to the migration guide.) It's the same trap as `c[c > 2][0] = 99` in [section C.5](C-numpy-in-miniature.md#5-fancy-indexing-and-boolean-masks-always-copy): two pairs of brackets, the first one builds a new object, the second writes into it, and the result is thrown away. The difference is that pandas 3 now tells you. The correct version does the selection and the assignment in *one* indexing operation on the original:

```python
df.loc[df["temp"] > 10, "temp"] = 0
print(df)
```

```text
   city  temp
0  Rome     0
1  Lima     0
2  Rome     0
3  Oslo     5
```

The other side of the same rule: a column you take out of a DataFrame is now independent.

```python
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
col = df["temp"]
col.iloc[0] = 99
print(df["temp"].tolist())
```

```text
[20, 18, 24, 5]
```

In pandas 2 this could have modified `df`. In pandas 3 it never does.

That doesn't mean every selection copies all the data, and this is the clever part. The copy is **lazy**: the selection shares memory with the original until somebody writes to one of them, and only then is the data actually copied.

```python
import numpy as np

df = pd.DataFrame({"x": [1, 2, 3]})
col = df["x"]
print(np.shares_memory(col.to_numpy(), df["x"].to_numpy()))
col.iloc[0] = 9
print(np.shares_memory(col.to_numpy(), df["x"].to_numpy()), df["x"].tolist())
```

```text
True
False [1, 2, 3]
```

Shared until the write, then separate. The miniature implements the same idea with a buffer that counts how many Series are reading it, and a check before every write:

```python
class Buffer:
    def __init__(self, data):
        self.data = list(data)
        self.readers = 0


class MiniSeries:
    def __init__(self, values, index=None, name=None, _buffer=None):
        if _buffer is None:
            _buffer = Buffer(values)
        self._buf = _buffer
        self._buf.readers += 1
        ...

    def _share(self):
        return MiniSeries(None, self.index, self.name, _buffer=self._buf)

    def _before_write(self):
        if self._buf.readers > 1:            # someone else can see this buffer
            self._buf.readers -= 1
            self._buf = Buffer(self._buf.data)   # so copy it before writing
            self._buf.readers = 1

    def set_loc(self, label, value):
        self._before_write()
        self._buf.data[self._position(label)] = value
```

```python
mdf = MiniDataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
col = mdf["temp"]
print(col._buf is mdf._cols["temp"]._buf)
col.set_loc(0, 99)
print(col._buf is mdf._cols["temp"]._buf, col.values, mdf["temp"].values)
```

```text
True
False [99, 18, 24, 5] [20, 18, 24, 5]
```

Real pandas does the same with more care: it tracks the objects that share each block of memory using weak references (an internal class called `BlockValuesRefs`), so when a sharing object is deleted, the others stop paying for a copy they no longer need. Our counter never goes down, which means it sometimes copies when pandas wouldn't. The visible behavior, that writes never leak between objects, is the same, and the test checks it.

> [!NOTE]
> If you have pandas code that relied on modifying a DataFrame through a selection, it stopped working in pandas 3.0: with the warning above in the chained case, and silently in the case of a column taken out and modified. Search your code for two consecutive pairs of brackets on the left of an `=`, and turn each one into a single `.loc[rows, columns] = value`.

## 7. groupby is split, apply, combine

`groupby` is another operation that's easy to use for years without quite knowing what it does. It's three steps, named in a paper that's worth reading for the idea alone ([Wickham, "The Split-Apply-Combine Strategy for Data Analysis", 2011](https://doi.org/10.18637/jss.v040.i01)<sup>\[11\]</sup>):

1. **Split** the rows into groups by the value of the key;
2. **Apply** a function to each group separately;
3. **Combine** the results into a new object, indexed by the group keys.

The miniature does each step in a few lines:

```python
class MiniGroupBy:
    def __init__(self, frame, key, sort=True):
        self.frame, self.key, self.sort = frame, key, sort

    def _split(self):
        groups = {}
        for pos, k in enumerate(self.frame._cols[self.key].values):
            groups.setdefault(k, []).append(pos)
        keys = sorted(groups) if self.sort else list(groups)
        return keys, groups

    def agg(self, column, func):
        keys, groups = self._split()
        values = self.frame._cols[column].values
        out = [func([values[p] for p in groups[k]]) for k in keys]   # apply
        return MiniSeries(out, keys, column)                         # combine

    def mean(self, column):
        return self.agg(column, lambda xs: sum(xs) / len(xs))
```

```python
g = mdf.groupby("city")
print(g._split())
print(g.mean("temp").index, g.mean("temp").values)
```

```text
(['Lima', 'Oslo', 'Rome'], {'Rome': [0, 2], 'Lima': [1], 'Oslo': [3]})
['Lima', 'Oslo', 'Rome'] [18.0, 5.0, 22.0]
```

The split step is just a dictionary from each key to the row positions that have it. Rome's rows are 0 and 2, so its mean is (20 + 24) / 2 = 22. And pandas, on the same data:

```python
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
print(df.groupby("city")["temp"].mean())
print(df.groupby("city", sort=False)["temp"].mean())
```

```text
city
Lima    18.0
Oslo     5.0
Rome    22.0
Name: temp, dtype: float64
city
Rome    22.0
Lima    18.0
Oslo     5.0
Name: temp, dtype: float64
```

Two details that the miniature makes visible. First, the result is indexed by the group keys, which is why a groupby result aligns with other things grouped by the same key, and why you often see `.reset_index()` after a groupby to turn the keys back into a column. Second, pandas **sorts the keys by default**. If you need them in order of first appearance (because the order means something, like time), pass `sort=False`, which is also faster on large data.

Once you see `groupby` as split-apply-combine, the rest of its API sorts itself out. `agg` applies a function that reduces each group to one value (mean, sum, count). `transform` applies a function that returns one value per row, so the result has the same index as the original and can be assigned back as a new column (like each row's difference from its group's mean). `apply` is the general case, and the slowest, because pandas can't know in advance what shape your function returns.

## 8. merge is a join, and joins can multiply your rows

Some of the most expensive pandas bugs don't raise anything. Here's a classic: you have a table of orders and you want to attach each customer's email:

```python
orders = pd.DataFrame({"customer": ["ana", "ana", "bo"], "amount": [10, 20, 5]})
emails = pd.DataFrame({"customer": ["ana", "ana", "bo"], "email": ["a@x", "ana@y", "b@x"]})
m = orders.merge(emails, on="customer")
print(len(orders), len(m))
print(m)
print(orders["amount"].sum(), m["amount"].sum())
```

```text
3 5
  customer  amount  email
0      ana      10    a@x
1      ana      10  ana@y
2      ana      20    a@x
3      ana      20  ana@y
4       bo       5    b@x
35 65
```

`ana` appears twice in the emails table (someone has two addresses), so each of her orders matched twice. Three orders became five rows, and total revenue went from 35 to 65. Nothing crashed. A model trained on `m` sees Ana's orders twice, and a dashboard built on it reports almost double the revenue.

The fix is to tell pandas what kind of join you think you're doing, so it can check:

```python
orders.merge(emails, on="customer", validate="many_to_one")
```

```text
MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
```

> [!TIP]
> Pass `validate=` to every `merge` whose shape you care about (`"one_to_one"`, `"one_to_many"`, `"many_to_one"`), and check `len()` before and after. It's about the cheapest data test you can write.

## 9. Missing values are skipped, by default

One more convention that's worth knowing before it surprises you: pandas reductions **skip** missing values.

```python
s = pd.Series([1.0, None, 3.0])
print(s.dtype, s.isna().tolist(), s.sum(), s.mean())
```

```text
float64 [False, True, False] 4.0 2.0
```

The mean is 2.0, the mean of the two values that exist, not of three. NumPy would have returned `nan` for the same data. Both choices are defensible, but they're different, and if 40% of a column is missing, "the mean" in pandas is the mean of the 60% that isn't. `s.isna().mean()` gives you the fraction of missing values, and it's worth printing before any other statistic.

## 10. Why apply is slow, and what to use instead

One more thing that comes straight out of the data model, and it's about speed. `df.apply(f, axis=1)` looks like the pandas way of doing something to every row, and it shows up in a lot of tutorials. Here's what it costs, on a million rows:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1_000_000
df = pd.DataFrame({"price": rng.uniform(1, 100, n), "qty": rng.integers(1, 10, n),
                   "city": rng.choice(["Rome", "Lima", "Oslo", "Kyiv", "Pune"], n)})

df.apply(lambda r: r["price"] * r["qty"], axis=1)      # row by row
df["price"] * df["qty"]                                # whole columns
[p * q for p, q in zip(df["price"], df["qty"])]        # a plain Python loop
```

```text
apply(axis=1)        6.33 s
vectorized           2.2 ms
python loop zip      169 ms
```

Same result (I checked with `np.allclose`), and the vectorized version is close to three thousand times faster. Even the plain Python loop beats `apply` by a wide margin. The reason is the data model: a DataFrame is a dict of columns ([section D.5](#5-a-dataframe-is-a-dict-of-series-that-share-an-index)), not a list of rows. To give your function a row, `apply(axis=1)` has to build a brand new Series for every row, with its own index, gathering one value from each column, a million times. The vectorized version hands two NumPy arrays to C code, which is [section C.10](C-numpy-in-miniature.md#10-memory-layout-and-why-vectorized-code-is-fast) all over again.

So the rule of thumb: think in columns. Arithmetic, comparisons and `np.where` work on whole columns. Strings have the `.str` accessor (`df["city"].str.upper()`), dates have `.dt`, and conditional logic usually becomes `np.where` or `np.select`. When nothing vectorized exists, a loop over `zip` of the columns you need is still much faster than `apply` over rows.

The same data model explains one more cheap win. A text column with a few distinct values repeated many times stores every one of those strings:

```python
print(df["city"].memory_usage(deep=True) / 1e6, df["city"].astype("category").memory_usage(deep=True) / 1e6)
```

```text
53.0 1.0
```

53 MB as strings, 1 MB as a **categorical**, which stores the five distinct names once and a small integer code per row (here `int8`). For columns like country, status or product type, that's a big saving in memory, and it makes the "set of possible values" explicit, which is information pandas (and you) can use.

## 11. read_csv guesses, and you should check

The miniature never reads a file, but in real work a lot of DataFrames start with `read_csv`, and `read_csv` has to guess a dtype for every column from text. Its guesses are reasonable and fast, and they're wrong often enough that checking them is a habit worth having. A small file with the usual problems:

```python
import io
import pandas as pd

raw = """customer_id,zip,signup,amount,vip
000123,02134,2026-01-05,19.99,yes
000124,10001,2026-01-06,,no
000125,94105,2026-01-07,"1,200.50",yes
"""
df = pd.read_csv(io.StringIO(raw))
print(df)
print(df.dtypes)
```

```text
   customer_id    zip      signup    amount  vip
0          123   2134  2026-01-05     19.99  yes
1          124  10001  2026-01-06       NaN   no
2          125  94105  2026-01-07  1,200.50  yes
customer_id    int64
zip            int64
signup           str
amount           str
vip              str
dtype: object
```

Every single column is a little wrong, and not one of them raised a damn thing:

- `customer_id` and `zip` **lost their leading zeros**. They look like numbers, so they became `int64`, and "02134" (a Boston zip code) is now 2134. Identifiers aren't numbers, even when they're made of digits: you'll never add two zip codes, and you'll often need to join on them.
- `signup` is text. `read_csv` doesn't parse dates unless you ask.
- `amount` is text, because one value has a thousands separator, so the whole column can't be a float. Run `df["amount"].sum()` and you get `'19.991,200.50'`: the strings, glued together. `.mean()` at least raises an error ("Cannot perform reduction 'mean' with string dtype"). The column that most needed to be numeric is the one that isn't.
- `vip` is text, where you probably wanted booleans.

You tell `read_csv` what you know:

```python
df = pd.read_csv(io.StringIO(raw), dtype={"customer_id": "str", "zip": "str"},
                 parse_dates=["signup"], thousands=",")
df["vip"] = df["vip"].map({"yes": True, "no": False})
print(df.dtypes)
```

```text
customer_id               str
zip                       str
signup         datetime64[us]
amount                float64
vip                       bool
dtype: object
```

Now the zeros are back ("000123", "02134"), the amount is 1200.5, and the dates are dates. (The `[us]` means microsecond resolution; pandas 3 picks the resolution from the data instead of always using nanoseconds.)

Dates deserve their own paragraph, because they fail in the quietest ways. If one value in a column has a different format, `parse_dates` **gives up on the whole column without saying anything**: with `2026-01-05`, `05/01/2026` and `2026-01-07`, you get a `str` column back and no warning. `pd.to_datetime` on the same column is louder, and the error is helpful:

```text
ValueError: time data "05/01/2026" doesn't match format "%Y-%m-%d". You might want to try:
    - passing `format` if your strings have a consistent format;
    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;
    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this.
```

And here's the trap. Following the hint, `format="mixed", dayfirst=True` (because "05/01/2026" is the 5th of January in most of the world outside the US) gives:

```text
0   2026-05-01
1   2026-01-05
2   2026-07-01
Name: signup, dtype: datetime64[us]
```

The odd value is now right, and **the two ISO dates are wrong**: "2026-01-05" became the 1st of May. In mixed mode every value is parsed on its own, and `dayfirst=True` applies to all of them, including the ones that were unambiguous to a human. No error, three plausible dates, two of them wrong. The safe options are a strict format with `errors="coerce"`, which turns everything that doesn't match into `NaT` so you can count and inspect them:

```python
pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
```

```text
0   2026-01-05
1          NaT
2   2026-01-07
Name: signup, dtype: datetime64[us]
```

or fixing the source, which is usually a conversation with whoever produces the file.

The same goes for missing values: `read_csv` treats a list of strings as missing by default (19 of them in pandas 3.1, among them `""`, `"NA"`, `"n/a"`, `"NaN"`, `"null"` and `"None"`), but not, for example, `"-"`. A numeric column with a single `"-"` in it becomes a text column, and once again nothing complains.

> [!TIP]
> After every `read_csv` of a file you didn't create, run three lines: `df.dtypes`, `df.isna().mean()` and `df.head()`. Compare the dtypes with what the columns *mean*. Then write the `dtype=`, `parse_dates=` and `na_values=` arguments you need, and keep them in the code. They're documentation of what the file contains.

## 12. Checking the miniature against pandas

The test follows the same pattern as the NumPy one, using `pandas.testing.assert_series_equal`, which compares values, index, dtype and NaN positions:

- 400 random pairs of Series with overlapping string indexes, added with and without `fill_value`;
- 200 lookups by label and by position on shuffled integer indexes (the `s[0]` trap);
- 300 random `groupby` aggregations, mean and sum;
- 100 copy-on-write checks: write into a selected column, and verify that the DataFrame didn't change and the column did, exactly as in pandas.

```text
1700 checks against pandas 3.1.0: all passed
```

## 13. Reading the pandas docs with this in your head

The pandas user guide is long, and most of it becomes easy once you know alignment and copy-on-write. The pages worth reading first:

- **"Intro to data structures"**: Series and DataFrame, the index, and alignment, in the authors' words.
- **"Indexing and selecting data"**: `.loc`, `.iloc`, boolean indexing, and the difference between labels and positions.
- **"Copy-on-Write"**: the rules from [section D.6](#6-copy-on-write-the-change-that-broke-old-tutorials), the history, and how to migrate old code.
- **"Group by: split-apply-combine"**: `agg`, `transform`, `apply` and `filter`, with examples.
- **"Merge, join, concatenate and compare"**: joins, `validate`, and `indicator=True`, which adds a column telling you which side each row came from.

And the habit that saves the most time: when a result looks wrong, print `.index` before anything else.

## 14. Exercises

1. Without running it, predict the result of `pd.Series([1, 2], index=["a", "a"]) + pd.Series([10], index=["a"])`. (Duplicate labels make alignment a many-to-many join.) Then run it, and extend the miniature to match.
2. Add `transform(column, func)` to `MiniGroupBy`: one output per row, in the original row order. Check it against pandas.
3. Implement `MiniSeries.__mul__` and `sub` with `fill_value`, and add them to the random test.
4. Write a function that merges two DataFrames and raises an error if the number of rows changed unexpectedly. When is a change in row count *not* a bug?
5. Take a pandas tutorial written before 2024 and find every line that behaves differently under copy-on-write.
6. Write `read_customers(path)` for the file of [section D.11](#11-readcsv-guesses-and-you-should-check): explicit dtypes, dates parsed with a strict format and `errors="coerce"`, and an `assert` that fails if more than 1% of the dates didn't parse. Then feed it a file with a `"-"` in the amount column and see which line catches it.

**Where this shows up in No Shortcuts:** lesson 10.1, "pandas without the pain", goes much further (method chaining, categoricals, time series, performance), and 10.2 covers Polars and DuckDB, which are what you reach for when pandas gets too slow.

---

[← C. NumPy in miniature](C-numpy-in-miniature.md) · [Contents](../README.md#contents) · [E. scikit-learn in miniature →](E-scikit-learn-in-miniature.md)
