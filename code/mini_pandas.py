"""A miniature pandas: values plus an index, alignment by label, copy-on-write, and split-apply-combine.

Pure Python on purpose. Missing values are float("nan"), like pandas does for numeric data.
"""
import math

NAN = float("nan")


class Buffer:
    """A list that knows how many Series are reading it. Writing to a shared buffer copies it first."""

    def __init__(self, data):
        self.data = list(data)
        self.readers = 0


class MiniSeries:
    def __init__(self, values, index=None, name=None, _buffer=None):
        if _buffer is None:
            _buffer = Buffer(values)
        self._buf = _buffer
        self._buf.readers += 1
        self.index = list(index) if index is not None else list(range(len(self._buf.data)))
        self.name = name
        if len(self.index) != len(self._buf.data):
            raise ValueError("index and values must have the same length")

    # -- copy-on-write: share the buffer until somebody writes ------------------------------
    def _share(self, index=None, name=None):
        return MiniSeries(None, self.index if index is None else index,
                          self.name if name is None else name, _buffer=self._buf)

    def _before_write(self):
        if self._buf.readers > 1:            # someone else can see this buffer: copy it first
            self._buf.readers -= 1
            self._buf = Buffer(self._buf.data)
            self._buf.readers = 1

    @property
    def values(self):
        return list(self._buf.data)

    # -- label vs position -------------------------------------------------------------------
    def _position(self, label):
        try:
            return self.index.index(label)
        except ValueError:
            raise KeyError(label) from None

    def loc(self, label):
        return self._buf.data[self._position(label)]

    def iloc(self, i):
        return self._buf.data[i]

    def __getitem__(self, label):            # like pandas: [] means label
        return self.loc(label)

    def set_loc(self, label, value):
        self._before_write()
        self._buf.data[self._position(label)] = value

    # -- alignment ---------------------------------------------------------------------------
    def _aligned(self, other):
        if self.index == other.index:
            return self.index
        labels = set(self.index) | set(other.index)
        try:
            return sorted(labels)
        except TypeError:                    # labels that can't be compared: keep first appearance
            return list(dict.fromkeys(self.index + other.index))

    def _binary(self, other, op, fill_value=None):
        index = self._aligned(other)
        mine = dict(zip(self.index, self._buf.data))
        theirs = dict(zip(other.index, other._buf.data))
        out = []
        for label in index:
            a, b = mine.get(label, fill_value), theirs.get(label, fill_value)
            out.append(NAN if a is None or b is None else op(a, b))
        return MiniSeries(out, index, self.name if self.name == other.name else None)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def add(self, other, fill_value=None):
        return self._binary(other, lambda a, b: a + b, fill_value)

    def __repr__(self):
        rows = "\n".join(f"{label!s:>6}  {value}" for label, value in zip(self.index, self._buf.data))
        return f"{rows}\nname: {self.name}"


class MiniDataFrame:
    """A dict of columns that share one index."""

    def __init__(self, columns, index=None):
        n = len(next(iter(columns.values())))
        self.index = list(index) if index is not None else list(range(n))
        self._cols = {name: MiniSeries(vals, self.index, name) for name, vals in columns.items()}

    def __getitem__(self, name):             # a column comes out sharing its buffer
        return self._cols[name]._share()

    def groupby(self, key):
        return MiniGroupBy(self, key)


class MiniGroupBy:
    """Split the row positions by key, apply a function to each group, combine into a Series."""

    def __init__(self, frame, key, sort=True):
        self.frame, self.key, self.sort = frame, key, sort

    def _split(self):
        groups = {}
        for pos, k in enumerate(self.frame._cols[self.key].values):
            groups.setdefault(k, []).append(pos)          # first-appearance order
        keys = sorted(groups) if self.sort else list(groups)
        return keys, groups

    def agg(self, column, func):
        keys, groups = self._split()
        values = self.frame._cols[column].values
        out = [func([values[p] for p in groups[k]]) for k in keys]   # apply, then combine
        return MiniSeries(out, keys, column)

    def mean(self, column):
        return self.agg(column, lambda xs: sum(xs) / len(xs))

    def sum(self, column):
        return self.agg(column, sum)


def isnan(x):
    return isinstance(x, float) and math.isnan(x)
