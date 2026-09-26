# %% [markdown]
# # Lab 01.1: The data model, for real
#
# Four small pieces from the lesson, implemented and checked:
#
# 1. A `Vec` class driven entirely by dunder methods.
# 2. A lazy `sliding_window` generator that works on infinite input.
# 3. A `set_seed` context manager that restores the global random state.
# 4. A `retry` decorator that retries only the errors you ask for.
#
# Every section ends with assertions. After it passes, break things on purpose (suggestions at the end).

# %%
from __future__ import annotations

import functools
import math
import random
import time
from collections import deque
from contextlib import contextmanager
from itertools import count, islice

# %% [markdown]
# ## 1. Vec

# %%
class Vec:
    __slots__ = ("_xs",)

    def __init__(self, *xs: float) -> None:
        self._xs = tuple(float(x) for x in xs)

    def __repr__(self) -> str:
        return f"Vec{self._xs}"

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, i: int) -> float:
        return self._xs[i]

    def __iter__(self):
        return iter(self._xs)

    def _check(self, other: Vec) -> None:
        if len(self) != len(other):
            raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")

    def __add__(self, other: Vec) -> Vec:
        self._check(other)
        return Vec(*(a + b for a, b in zip(self, other)))

    def __sub__(self, other: Vec) -> Vec:
        self._check(other)
        return Vec(*(a - b for a, b in zip(self, other)))

    def __neg__(self) -> Vec:
        return Vec(*(-a for a in self))

    def __mul__(self, k: float) -> Vec:
        return Vec(*(k * a for a in self))

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> Vec:
        return Vec(*(a / k for a in self))

    def __matmul__(self, other: Vec) -> float:
        self._check(other)
        return sum(a * b for a, b in zip(self, other))

    def __abs__(self) -> float:
        return math.sqrt(self @ self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec) and self._xs == other._xs

    def __hash__(self) -> int:
        return hash(self._xs)

    def __bool__(self) -> bool:
        return any(self._xs)

    def normalized(self) -> Vec:
        """Unit vector. Raises for the zero vector: it has no direction, and silently
        returning zeros would hide the bug that produced it."""
        n = abs(self)
        if n == 0:
            raise ValueError("cannot normalize the zero vector")
        return self / n


v, w = Vec(1, 2), Vec(3, 4)
assert v + w == Vec(4, 6)
assert 3 * v == v * 3 == Vec(3, 6)          # __rmul__ at work
assert v @ w == 11
assert abs(w) == 5
assert -v == Vec(-1, -2)
assert len({v, Vec(1, 2), w}) == 2          # equal objects hash equally
assert not Vec(0, 0) and Vec(0, 1)          # __bool__
assert math.isclose(abs(w.normalized()), 1.0)
try:
    Vec(1, 2) + Vec(1, 2, 3)
    raise AssertionError("dimension mismatch should raise")
except ValueError:
    pass
try:
    v.oops = 1                              # __slots__: typos in attribute names fail loudly
    raise AssertionError("__slots__ should prevent new attributes")
except AttributeError:
    pass
print("Vec ok:", v, w, v @ w)

# %% [markdown]
# ## 2. sliding_window
#
# It must work on an infinite iterator, which means it can never call `len()` or `list()` on its input.

# %%
def sliding_window(iterable, n: int):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


assert list(sliding_window([1, 2, 3, 4], 2)) == [(1, 2), (2, 3), (3, 4)]
assert list(sliding_window([1, 2], 3)) == []                     # not enough elements: nothing
assert list(islice(sliding_window(count(), 3), 3)) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]  # infinite input

# The exhausted-generator trap from the lesson, demonstrated:
g = sliding_window("abc", 2)
assert len(list(g)) == 2
assert list(g) == []                                             # second pass: silently empty
print("sliding_window ok")

# %% [markdown]
# ## 3. set_seed
#
# Inside the block, randomness is deterministic. Outside, the program's random stream continues exactly as if the
# block had never run.

# %%
@contextmanager
def set_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


random.seed(123)
expected = [random.random() for _ in range(3)]     # what the outer stream produces with no interruption

random.seed(123)
outer = [random.random()]
with set_seed(0):
    inner_a = [random.random() for _ in range(5)]
with set_seed(0):
    inner_b = [random.random() for _ in range(5)]
outer += [random.random() for _ in range(2)]

assert inner_a == inner_b, "same seed inside the block must give the same numbers"
assert outer == expected, "the outer random stream must be unaffected by the block"
print("set_seed ok")

# %% [markdown]
# ## 4. retry
#
# We simulate a flaky service that fails twice with a ConnectionError and then succeeds, and a buggy function
# that raises a ValueError. The decorator must retry the first and give up immediately on the second, because
# retrying a bug just makes it slower.

# %%
def retry(times: int = 3, delay: float = 0.0, exceptions: tuple[type[BaseException], ...] = (Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
                    time.sleep(delay * 2**attempt)
        return wrapper
    return decorator


calls = {"flaky": 0, "buggy": 0}


@retry(times=5, exceptions=(ConnectionError,))
def flaky_service() -> str:
    """Pretend network call."""
    calls["flaky"] += 1
    if calls["flaky"] < 3:
        raise ConnectionError("connection reset by peer")
    return "features"


@retry(times=5, exceptions=(ConnectionError,))
def buggy() -> None:
    calls["buggy"] += 1
    raise ValueError("this is a bug, not a network problem")


assert flaky_service() == "features" and calls["flaky"] == 3
try:
    buggy()
except ValueError:
    pass
assert calls["buggy"] == 1, "a ValueError must not be retried"
assert flaky_service.__name__ == "flaky_service" and "network" in flaky_service.__doc__  # functools.wraps
print("retry ok")

# %% [markdown]
# ## Break it on purpose
#
# - Remove `__hash__` from `Vec`. What happens to `{v, w}`? (Python sets `__hash__ = None` when you define
#   `__eq__` alone. Read the error.)
# - Remove `__rmul__`. Which assertion fails, and with what message?
# - In `set_seed`, drop the `try/finally` and raise an exception inside the `with` block. Is the state restored?
# - Change `retry` to catch `Exception` for `buggy`. Count the calls.

# %%
print("\nAll checks passed.")
