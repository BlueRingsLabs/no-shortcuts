# C. NumPy in miniature

Everything in this addendum sits on NumPy. pandas columns are NumPy arrays (or close relatives), scikit-learn converts whatever you pass it into NumPy arrays before doing anything, and PyTorch tensors copy NumPy's data model so closely that most of this chapter applies to them unchanged. If you understand one thing about NumPy really well, it should be this chapter.

## 1. An array is a buffer and four pieces of metadata

A NumPy array prints like a nested list, but underneath it's a single flat block of memory (the **buffer**) plus a few pieces of metadata that tell NumPy how to read it:

- **dtype**: what each element is and how many bytes it takes (`int64` is 8 bytes, `float32` is 4);
- **shape**: how many elements along each axis;
- **strides**: how many bytes to jump in memory to move one step along each axis;
- **offset**: where in the buffer the first element is.

```python
import numpy as np

a = np.arange(12, dtype=np.int64).reshape(3, 4)
print(a.shape, a.dtype, a.strides)
```

```text
(3, 4) int64 (32, 8)
```

Read the strides as a sentence: to move one step down a column (axis 0) jump 32 bytes, which is a whole row of four 8-byte integers; to move one step along a row (axis 1) jump 8 bytes, one integer. The element at row `i`, column `j` lives at byte

```text
offset + i * 32 + j * 8
```

That formula is the whole of NumPy's data model. Everything else in this chapter (slicing, transposing, reshaping, broadcasting, and the bugs they cause) is a different way of choosing shape, strides and offset over the same buffer.

## 2. Building MiniArray

Let's build it. Our miniature keeps the buffer as a flat Python list and counts strides in elements instead of bytes, which is the same thing divided by the item size.

```python
from itertools import product


def c_strides(shape):
    strides, step = [], 1
    for n in reversed(shape):
        strides.append(step)
        step *= n
    return tuple(reversed(strides))


class MiniArray:
    def __init__(self, buffer, shape, strides=None, offset=0):
        self.buffer = buffer                    # a flat list, shared between views
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else c_strides(shape)
        self.offset = offset

    @classmethod
    def arange(cls, n):
        return cls(list(range(n)), (n,))

    def _flat_index(self, idx):
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    def item(self, *idx):
        return self.buffer[self._flat_index(idx)]
```

`c_strides` computes the default layout: for shape `(3, 4)` the last axis has stride 1, and the first has stride 4, because skipping one row means skipping four elements. NumPy calls this layout **C order** (row-major), as opposed to **Fortran order**, where the first axis moves fastest.

Reading one element is the dot product of the index with the strides, plus the offset:

```python
m = MiniArray.arange(12).reshape(3, 4)   # reshape comes in section C.4
print(m.shape, m.strides, m.offset)
print(m.item(1, 2), a[1, 2])
```

```text
(3, 4) (4, 1) 0
6 6
```

Same answer as NumPy, and the strides `(4, 1)` are NumPy's `(32, 8)` divided by 8 bytes.

## 3. Slicing is arithmetic on strides

When you slice an array, NumPy doesn't copy anything. It computes a new shape, new strides and a new offset, and points them at the same buffer. The result is called a **view**.

For each axis, an integer index removes the axis and moves the offset; a slice `start:stop:step` keeps the axis, moves the offset to `start`, sets the new length, and multiplies the stride by `step`:

```python
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        offset, shape, strides = self.offset, [], []
        for k, n, s in zip(key, self.shape, self.strides):
            if isinstance(k, int):
                if k < 0:
                    k += n
                if not 0 <= k < n:
                    raise IndexError(f"index {k} out of range for axis of size {n}")
                offset += k * s
            else:
                start, stop, step = k.indices(n)
                offset += start * s
                shape.append(len(range(start, stop, step)))
                strides.append(s * step)
        return MiniArray(self.buffer, shape, strides, offset)
```

`slice.indices(n)` is a built-in that turns a slice like `1:3` or `::-1` into concrete `(start, stop, step)` for a sequence of length `n`, negative steps included. That's the only Python trivia in this function.

Let's take the middle two columns:

```python
mv = m[:, 1:3]
print(mv.shape, mv.strides, mv.offset, mv.tolist())
```

```text
(3, 2) (4, 1) 1 [[1, 2], [5, 6], [9, 10]]
```

Same strides as the original, offset 1 (the view starts at element 1 of the buffer), shape `(3, 2)`. Nothing was copied. NumPy does exactly the same:

```python
av = a[:, 1:3]
print(av.strides, np.shares_memory(a, av))
av[0, 0] = 100
print(a[0])
```

```text
(32, 8) True
[  0 100   2   3]
```

Writing into the slice wrote into the original, because they're the same memory.

> [!WARNING]
> A slice is a view. If you slice an array, change the slice, and expect the original to stay the same, it won't. When you need an independent piece, say so: `a[:, 1:3].copy()`. This is a classic source of "my training data changed by itself" bugs, especially when a function receives a slice and modifies its argument.

Negative steps work the same way: `a[::-1]` is a view with a negative stride and an offset pointing at the last row. Here's a slice with both:

```python
s = a[::2, ::-1]
print(s.strides, s.T.strides)
```

```text
(64, -8) (-8, 64)
```

Every second row (stride doubled to 64 bytes), columns reversed (stride −8). Still no copy.

## 4. Transpose and reshape, and when NumPy copies

Transposing is the easiest operation in NumPy: reverse the shape and reverse the strides. No data moves.

```python
    @property
    def T(self):
        return MiniArray(self.buffer, self.shape[::-1], self.strides[::-1], self.offset)
```

```python
print(m.T.shape, m.T.strides, a.T.strides, a.T.flags["C_CONTIGUOUS"])
```

```text
(4, 3) (1, 4) (8, 32) False
```

The transposed array is no longer **C-contiguous**: walking it in row-major order no longer walks the buffer in order. That's fine for reading. It matters for reshape, because reshape has to answer a question: can the new shape be expressed as new strides over the same buffer?

If the array is contiguous, yes, always: reinterpret the same elements with the default strides of the new shape. If it isn't, sometimes no, and then NumPy makes a copy. Our miniature does the simple version of that rule (contiguous means view, anything else means copy; real NumPy is a bit smarter and finds a view in some non-contiguous cases too):

```python
    def is_c_contiguous(self):
        return self.strides == c_strides(self.shape)

    def reshape(self, *shape):
        if _prod(self.shape) != _prod(shape):
            raise ValueError(f"cannot reshape array of size {_prod(self.shape)} into shape {shape}")
        if self.is_c_contiguous():
            return MiniArray(self.buffer, shape, c_strides(shape), self.offset)
        return MiniArray(self.tolist_flat(), shape)
```

(`_prod` multiplies a tuple of numbers, and `tolist_flat` reads every element in row-major order using `item`. The complete file is in the companion code.)

In real NumPy:

```python
r1 = a.reshape(2, 6)
r2 = a.T.reshape(12)
print(np.shares_memory(a, r1), np.shares_memory(a, r2))
```

```text
True False
```

`r1` is a view; `r2` is a copy, because the transposed elements are not in buffer order. Modern NumPy's `reshape` takes a `copy` argument (it's in the reference from NumPy 2.1 on, and NumPy 1.x doesn't have it), so you can say "a view or nothing":

```python
a.T.reshape(12, copy=False)
```

```text
ValueError: Unable to avoid creating a copy while reshaping.
```

> [!NOTE]
> NumPy 2.5 deprecates changing an array's shape by assigning to it (`x.shape = (12,)`), which used to be the classic way of asking for a view-only reshape. Use `reshape(..., copy=False)` instead.

So: `reshape`, `ravel` and `T` *may* give you a view; `flatten` and `copy` always give you a copy. When it matters, don't guess, ask: `np.shares_memory(x, y)` tells you whether two arrays overlap in memory.

## 5. Fancy indexing and boolean masks always copy

There's a second kind of indexing, and it follows different rules. Indexing with a list or an array of integers (**fancy indexing**) or with a boolean array (a **mask**) can pick elements in any order, with repetitions. No combination of shape, strides and offset can express that, so NumPy has to copy:

```python
f = a[[0, 2]]
print(np.shares_memory(a, f))
```

```text
False
```

That explains one of the most confusing behaviors in NumPy. These two lines look like they should do the same thing:

```python
b = np.arange(6)
b[b > 2] = 0
print(b)

c = np.arange(6)
c[c > 2][0] = 99
print(c)
```

```text
[0 1 2 0 0 0]
[0 1 2 3 4 5]
```

The first works, the second silently does nothing. The reason is that Python turns them into different method calls. `b[mask] = 0` is a single `b.__setitem__(mask, 0)`, which writes into `b`. `c[mask][0] = 99` is `c.__getitem__(mask)` first, which builds a **copy**, and then `__setitem__` on that copy, which is thrown away immediately.

> [!TIP]
> If an assignment "does nothing", count the brackets. More than one pair of indexing brackets on the left side of `=`, with a mask or a list in any but the last pair, means you're writing into a temporary copy. pandas has the same trap, and [chapter D](D-pandas-in-miniature.md) shows what pandas 3.0 did about it.

## 6. Broadcasting is a stride of zero

Broadcasting is how NumPy combines arrays of different shapes: a `(3, 1)` column plus a `(4,)` row gives a `(3, 4)` table. It's usually taught as a rule to memorize, and it's easier to understand as one more trick with strides.

First, the rule for the shapes. Align them on the right; compare sizes axis by axis; each pair must be equal, or one of them must be 1. Missing axes on the left count as 1.

```python
def broadcast_shape(*shapes):
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out = []
    for sizes in zip(*padded):
        others = {n for n in sizes if n != 1}
        if len(others) > 1:
            raise ValueError(f"shapes {shapes} are not broadcastable")
        out.append(others.pop() if others else 1)
    return tuple(out)
```

```python
print(broadcast_shape((3, 1), (4,)), np.broadcast_shapes((3, 1), (4,)))
```

```text
(3, 4) (3, 4)
```

Now the trick. How do you make a `(3, 1)` column look like a `(3, 4)` table without copying it four times? Give the stretched axis **stride 0**. Moving along that axis then moves zero bytes in memory, so the same element is read again and again:

```python
def broadcast_to(a, shape):
    pad = len(shape) - len(a.shape)
    strides = [0] * pad
    for n_old, s_old, n_new in zip(a.shape, a.strides, shape[pad:]):
        if n_old == n_new:
            strides.append(s_old)
        elif n_old == 1:
            strides.append(0)                # stretched axis: stride 0
        else:
            raise ValueError(f"cannot broadcast {a.shape} to {shape}")
    return MiniArray(a.buffer, shape, strides, a.offset)


def add(a, b):
    shape = broadcast_shape(a.shape, b.shape)
    a2, b2 = broadcast_to(a, shape), broadcast_to(b, shape)
    out = [a2.item(*i) + b2.item(*i) for i in product(*(range(n) for n in shape))]
    return MiniArray(out, shape)
```

```python
col = MiniArray([10, 20, 30], (3, 1))
row = MiniArray.arange(4)
print(add(col, row).tolist())
print(np.array([[10], [20], [30]]) + np.arange(4))
```

```text
[[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
[[10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]]
```

NumPy does literally this. You can see the zero stride yourself:

```python
bt = np.broadcast_to(np.arange(4), (3, 4))
print(bt.strides, bt.flags.writeable)
```

```text
(0, 8) False
```

Stride 0 along the rows, and the view is read-only: writing to one "copy" of the row would write to all of them, because there's only one.

When the shapes don't fit, the error message reports exactly the comparison that failed:

```python
np.ones((3, 2)) + np.ones(3)
```

```text
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```

Aligned on the right, 2 and 3 are neither equal nor 1.

## 7. A broadcasting bug that doesn't crash

Here's a mistake I'd bet a lot of you have made, or will. You want to center each row of a matrix (subtract each row's mean from that row):

```python
X = np.random.default_rng(0).normal(size=(5, 3))
X - X.mean(axis=1)
```

```text
ValueError: operands could not be broadcast together with shapes (5,3) (5,)
```

`X.mean(axis=1)` has shape `(5,)`, one mean per row. Aligned on the right, `(5,)` is compared with the *columns* of `X`, which are 3. The fix is to keep the reduced axis as a size-1 axis, so the means line up with the rows:

```python
(X - X.mean(axis=1, keepdims=True)).mean(axis=1).round(12)
```

```text
[-0.  0.  0.  0.  0.]
```

Every row now has mean zero. So far, NumPy protected you with an error. Now look at the same bug with a square matrix:

```python
X = np.array([[1., 2., 3.], [10., 20., 30.], [100., 200., 300.]])
print(X.mean(axis=1))
print(X - X.mean(axis=1))                  # wrong, and no error
print(X - X.mean(axis=1, keepdims=True))   # right
```

```text
[  2.  20. 200.]
[[  -1.  -18. -197.]
 [   8.    0. -170.]
 [  98.  180.  100.]]
[[  -1.    0.    1.]
 [ -10.    0.   10.]
 [-100.    0.  100.]]
```

With three rows and three columns, `(3,)` broadcasts against the columns without complaint, and you subtract row 0's mean from column 0, row 1's mean from column 1, and so on. The result is garbage with the right shape. This is the kind of bug that survives into a model, because nothing crashes.

> [!TIP]
> After any reduction you plan to broadcast back (`mean`, `sum`, `max`, `std` along an axis), use `keepdims=True`. It costs nothing, and it makes the broadcast line up with the axis you reduced, whatever the shape of the data.

## 8. einsum is just named loops

`np.einsum` looks like magic notation, and it's the most direct way of writing array operations once you see what it is: **every letter is a loop, and letters that don't appear in the output are summed over**. `"ij,jk->ik"` reads as "for every i, j and k, multiply `x[i, j]` by `y[j, k]`, and add the products into `out[i, k]`", which is matrix multiplication.

The miniature writes exactly that:

```python
def mini_einsum(spec, x, y):
    inputs, out = spec.replace(" ", "").split("->")
    sx, sy = inputs.split(",")
    sizes = {}
    for letters, arr in ((sx, x), (sy, y)):
        for letter, n in zip(letters, arr.shape):
            if sizes.setdefault(letter, n) != n:
                raise ValueError(f"size mismatch for index {letter!r}")
    letters = sorted(sizes)
    result = {}
    for values in product(*(range(sizes[l]) for l in letters)):
        env = dict(zip(letters, values))
        key = tuple(env[l] for l in out)
        prod_ = x.item(*(env[l] for l in sx)) * y.item(*(env[l] for l in sy))
        result[key] = result.get(key, 0) + prod_                    # missing letters get summed
    shape = tuple(sizes[l] for l in out)
    flat = [result.get(idx, 0) for idx in product(*(range(n) for n in shape))]
    return MiniArray(flat, shape)
```

With that reading, a whole family of operations becomes one line each:

| Spec | Meaning |
|---|---|
| `"ij,jk->ik"` | matrix product |
| `"ij,ij->i"` | row-wise dot product |
| `"i,j->ij"` | outer product |
| `"bij,bjk->bik"` | a batch of matrix products (attention does this) |
| `"ij,jk->"` | the sum of all the entries of a matrix product |

The miniature gives the same results as `np.einsum` on all five. The real one is far faster, because it hands the loops to optimized code and can reorder them (pass `optimize=True` when you chain more than two operands), but it computes the same thing.

## 9. dtypes, or what happens when a number doesn't fit

The dtype is the easiest part of the data model to ignore, until it bites, and it bites quietly. A NumPy array has one dtype for every element, fixed, and arithmetic stays inside it. For integers that means something that surprises people coming from plain Python, whose integers never overflow:

```python
import numpy as np

a = np.array([100, 120], dtype=np.int8)
print(a + a, (a + a).dtype)
```

```text
[-56 -16] int8
```

An `int8` holds −128 to 127. 100 + 100 = 200 doesn't fit, so it wraps around to −56, and NumPy says nothing. The docstring of `np.sum` puts it plainly: arithmetic is modular when using integer types, and no error is raised on overflow. The same happens with `int32` (2,147,483,647 + 1 gives −2,147,483,648), which was the default integer on Windows before NumPy 2.

NumPy 2 changed the rules here (NEP 50), so older code and older answers on the internet can be wrong about this part. When you mix an array with a plain Python number, the Python number is "weak": it adopts the array's dtype instead of upgrading it.

```python
print((np.array([1, 2], dtype=np.int8) + 1).dtype)
print((np.array([1, 2], dtype=np.int8) + 1.5).dtype)
print((np.float32(3.0) + 3.0).dtype, (np.float32(3.0) + np.float64(3.0)).dtype)
```

```text
int8
float64
float32 float64
```

An `int8` array plus the Python integer `1` stays `int8`. A float32 plus a Python float stays float32, but a float32 plus a NumPy float64 becomes float64. And if the Python number doesn't even fit in the array's dtype, NumPy 2 refuses instead of guessing:

```python
np.int8(100) + 300
```

```text
OverflowError: Python integer 300 out of bounds for int8
```

(For the record, `np.int8(100) + 100` doesn't raise: it gives −56 with a `RuntimeWarning` about overflow. Scalars warn, arrays don't. I didn't design it xD)

And one more surprise:

```python
print(np.result_type(np.int8, np.uint8), np.result_type(np.int64, np.uint64))
```

```text
int16 float64
```

There's no integer type that can hold every `int64` and every `uint64`, so NumPy goes to `float64`, which can't represent all of them exactly either. If you ever mix signed and unsigned 64-bit IDs, you can end up comparing floats that used to be different integers.

> [!WARNING]
> Before doing arithmetic on integer data that came from a file, an image or a database, check `x.dtype` and `np.iinfo(x.dtype)`. If the values can grow (sums, products, differences of timestamps), convert first: `x.astype(np.int64)` or `x.astype(np.float64)`.

Floats have their own version of the problem, which is precision. `float32` has about 7 significant digits:

```python
print(float(np.float32(0.1)))
```

```text
0.10000000149011612
```

Now add 0.1, stored as `float32`, two million times, the naive way, one at a time into a `float32` accumulator:

```python
x = np.full(10_000_000, 0.1, dtype=np.float32)
total = np.float32(0.0)
for v in x[:2_000_000]:
    total += v
print(total)
```

```text
197024.78
```

The right answer is about 200,000. The loop is off by about 1.5%, because once the total is large, each tiny 0.1 is rounded away in the addition. NumPy's own `sum`, on all ten million values, still in `float32`:

```python
print(x.sum(), x.sum(dtype=np.float64))
```

```text
1.0000001e+06 1000000.0149011612
```

Almost exact. The reason is in the same docstring: NumPy uses **pairwise summation**, adding numbers in a tree of small partial sums instead of one long running total, so the rounding errors stay small (the docs say this improved precision is always used when no `axis` is given). Passing `dtype=np.float64` accumulates in 64 bits, which gives you the exact sum of the `float32` values. The same concern is why deep learning frameworks accumulate some sums in higher precision even when the data is `float32` or lower.

## 10. Memory layout, and why vectorized code is fast

Everything in this chapter so far was about correctness. One section about speed, because "use vectorized NumPy, not loops" is advice that gets repeated much more often than it gets explained.

The numbers first, measured on a 4-core cloud machine, the best of several runs of each (yours will differ, the ratios less so):

```python
import timeit

x = np.random.default_rng(0).random(1_000_000)

def loop():
    s = 0.0
    for v in x:
        s += v * v
    return s
```

```text
python loop sum of squares   128 ms
numpy  (x * x).sum()        1.13 ms
numpy  x @ x                0.09 ms
```

The loop is about 110 times slower than `(x * x).sum()`, and about 1,400 times slower than `x @ x`. Same math. The reason comes straight out of [section C.1](#1-an-array-is-a-buffer-and-four-pieces-of-metadata). A Python loop handles one element at a time, and each element is a full Python object: the interpreter has to look up its type, unbox the number, do the multiplication, box the result into a new object, and do the same for the addition, a million times. `(x * x).sum()` walks the buffer in C, with no Python objects at all, using CPU instructions that handle several numbers at once. And `x @ x` goes further: it hands the whole dot product to a BLAS library, code tuned for exactly this operation on this kind of CPU, without even creating the temporary array `x * x`.

Layout matters too, because memory is read in blocks. Copying a big array is fast when you read it in the order it's stored, and slow when you jump around:

```python
a = np.random.default_rng(0).random((4000, 4000))
a.copy()                       # reads and writes in storage order
np.ascontiguousarray(a.T)      # reads with a stride of 4000 * 8 bytes
```

```text
copy C                   30.4 ms
copy of .T to C order   160.8 ms
```

Same amount of data, five times slower, only because the transposed array is read column by column, jumping 32,000 bytes between consecutive elements. (Sums along the two axes, on the other hand, cost about the same here, 12.5 ms and 11.5 ms, because NumPy is smart about the order in which it walks the buffer for reductions. Don't guess, measure.)

> [!TIP]
> If a hot loop in your code reads a transposed or strided array many times, make one contiguous copy first (`np.ascontiguousarray`). And when you vectorize, prefer the operation that says the most: `x @ x` over `(x * x).sum()`, `np.linalg.norm` over writing the square root yourself. The more of the problem you hand to NumPy at once, the more of it runs in optimized code.

## 11. Random numbers: one generator, passed around

Every test in this addendum uses random inputs, and every ML experiment uses randomness somewhere (splits, initialization, shuffling, dropout), so it's worth knowing how NumPy's randomness is designed. There are two APIs, and plenty of tutorials still teach the old one.

The old one is a single global generator, hidden inside the `np.random` module, which `np.random.seed` resets:

```python
np.random.seed(0)
print(np.random.rand(3).round(4))
```

```text
[0.5488 0.7152 0.6028]
```

The problem is the word "global". Anything that draws from it moves it forward, including code you didn't write. Seed it, call scikit-learn's `train_test_split` without a `random_state` (which uses the same global generator), and then draw again:

```python
from sklearn.model_selection import train_test_split

np.random.seed(0)
train_test_split(np.arange(10), shuffle=True)
print(np.random.rand(3).round(4))
```

```text
[0.4376 0.8918 0.9637]
```

Different numbers, from the same seed, because a library call in between consumed some of the stream. In a real project those calls are scattered across modules, so "I set the seed at the top" doesn't make anything reproducible: add a line anywhere, or change the order of two calls, and every random number after it changes.

The new API, since NumPy 1.17, makes the generator an object:

```python
rng = np.random.default_rng(0)
print(type(rng).__name__, type(rng.bit_generator).__name__, rng.random(3).round(4))
```

```text
Generator PCG64 [0.637  0.2698 0.041 ]
```

A `Generator` (here using the PCG64 algorithm underneath) has its own state, and only code you give it to can draw from it. Same seed, same calls, same numbers; and other code drawing from *its* generator can't disturb yours. The pattern is: create one generator at the top of your program from one seed, and **pass it** to whatever needs randomness. When you need several independent streams (one per worker process, one per experiment), don't invent seeds like 0, 1, 2; ask the generator for independent children with `rng.spawn(n)`, which NumPy designed for exactly that.

Two things to know before you rely on it:

- The other libraries have their own generators. `np.random.seed` does nothing to PyTorch, which has its own (`torch.manual_seed`, and the `torch.Generator` objects we used in [section F.8](F-pytorch-in-miniature.md#8-dataset-and-dataloader)). scikit-learn's `random_state` takes an integer or an old-style `RandomState`; pass it a new `Generator` and scikit-learn 1.9 refuses with an `InvalidParameterError`. The usual bridge is an integer drawn from your generator: `random_state=int(rng.integers(2**32))`.
- "Reproducible" has limits. NumPy's compatibility policy promises the same stream for the same seed and the same calls "on the same build of numpy, in the same environment, on the same machine", and says that improvements to the algorithms can change the streams in feature releases. It even notes that `rng.random()` called five times isn't guaranteed to give the same numbers as `rng.random(5)`. Across machines and versions, reproducibility means recording the seed, the versions and the code.

## 12. Checking the miniature against the real thing

A miniature is only worth something if it agrees with the real library. Testing it on two or three examples you picked yourself proves very little, because you pick the examples you already understand. So the test generates hundreds of random cases and compares every result with NumPy:

- random shapes with 1 to 3 axes, random mixes of integers and slices, steps of 1, 2, 3, −1 and −2: same shape, same values, and the same strides (once multiplied by the item size);
- transposes of every one of those;
- random pairs of shapes for broadcasting, including pairs that must fail: same result shape, same error, same values after `add`;
- the five `einsum` specs above.

```text
805 checks against NumPy 2.5.4: all passed
```

This is **property-based testing** done by hand: instead of checking examples, you check that a property (here, "agrees with NumPy") holds over many random inputs. The library [Hypothesis](https://hypothesis.readthedocs.io/)<sup>\[8\]</sup> does it properly, including shrinking a failing case to the smallest input that still fails. It's worth learning on its own, because it's also the best way to test your own data-processing code.

Try breaking the miniature on purpose to see the test do its job: change `offset += start * s` to `offset += start` in `__getitem__`, run the test again, and look at which cases fail first. They're the ones where a slice starts somewhere other than the beginning of an axis with a stride bigger than 1, which is exactly what that line is for.

## 13. Reading the NumPy docs with this in your head

Now the documentation reads differently. A few places worth going to directly:

- The user guide pages on copies and views and on broadcasting ([numpy.org/doc/stable/user/basics.copies.html](https://numpy.org/doc/stable/user/basics.copies.html)<sup>\[9\]</sup> and its neighbor, "Broadcasting"). After this chapter they're short and obvious, which is the sign you've understood them.
- Any function's docstring, with `help(np.mean)` or `np.mean?` in Jupyter. Look for the parameters that come back everywhere: `axis` (which axis to reduce), `keepdims` (keep it as size 1), `out` (write into an existing array instead of allocating), `dtype` (accumulate in a different type, which matters when you sum a lot of `float32` values). Once you know these, most of the reference reads itself.
- Universal functions (ufuncs): `np.add`, `np.exp`, `np.maximum` and friends are all ufuncs, and they share methods: `np.add.reduce` is `sum`, `np.add.accumulate` is `cumsum`, `np.maximum.at` does indexed updates without the fancy-indexing copy trap. When you see a ufunc in the docs, you already know half of what it can do.
- The array's own attributes, when you debug: `x.shape`, `x.dtype`, `x.strides`, `x.flags`, `x.base` (the array this one is a view of, or `None`), and `np.shares_memory(x, y)`.

The original NumPy paper is also very readable, and it explains this data model in the authors' own words ([Harris et al., "Array programming with NumPy", *Nature*, 2020](https://doi.org/10.1038/s41586-020-2649-2)<sup>\[10\]</sup>).

## 14. Exercises

Do them before looking anything up, and predict the output before you run it.

1. Without running anything, write down the shape, strides and offset of `np.arange(24).reshape(2, 3, 4)[1, ::2, ::-1]` (with `int64`). Then check with NumPy.
2. Add `swapaxes(i, j)`, `expand_dims(axis)` and `squeeze()` to `MiniArray`. All three are pure stride operations, no copying.
3. Implement `sum(axis)` for `MiniArray` and add it to the random test.
4. `np.lib.stride_tricks.sliding_window_view(np.arange(10), 3)` gives every window of length 3 without copying. What are its strides? Implement it in the miniature.
5. Find a non-contiguous array that real NumPy can reshape *without* a copy, but our miniature copies. (Hint: think about which axes you merge.)
6. Write a script that seeds the global generator with `np.random.seed(0)` and prints a few random numbers at the end. Insert `np.random.permutation(5)` near the top and run it again. Then rewrite it with one `default_rng(0)` passed to each function that needs randomness, and repeat the experiment. Which version is still reproducible after the change?

**Where this shows up in No Shortcuts:** lesson 01.3, "NumPy and vectorization: thinking in arrays", is where the course turns this into speed, and every lab after it leans on broadcasting.

---

[← B. A method for learning any library](B-a-method-for-learning-any-library.md) · [Contents](../README.md#contents) · [D. pandas in miniature →](D-pandas-in-miniature.md)
