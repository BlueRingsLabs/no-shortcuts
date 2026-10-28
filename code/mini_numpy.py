"""A miniature NumPy: an array is a flat buffer plus shape, strides and an offset.

Everything here is pure Python on purpose. The point is to see the data model, not to be fast.
Strides are counted in elements (NumPy counts them in bytes; divide by itemsize to compare).
"""
from itertools import product


def c_strides(shape):
    strides, step = [], 1
    for n in reversed(shape):
        strides.append(step)
        step *= n
    return tuple(reversed(strides))


class MiniArray:
    def __init__(self, buffer, shape, strides=None, offset=0):
        self.buffer = buffer                    # a flat Python list, shared between views
        self.shape = tuple(shape)
        self.strides = tuple(strides) if strides is not None else c_strides(shape)
        self.offset = offset

    @classmethod
    def arange(cls, n):
        return cls(list(range(n)), (n,))

    # -- reading one element: the whole trick is this dot product ----------------------------
    def _flat_index(self, idx):
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    def item(self, *idx):
        return self.buffer[self._flat_index(idx)]

    # -- views: new shape/strides/offset, same buffer ----------------------------------------
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

    @property
    def T(self):
        return MiniArray(self.buffer, self.shape[::-1], self.strides[::-1], self.offset)

    def is_c_contiguous(self):
        return self.strides == c_strides(self.shape)

    def reshape(self, *shape):
        if _prod(self.shape) != _prod(shape):
            raise ValueError(f"cannot reshape array of size {_prod(self.shape)} into shape {shape}")
        if self.is_c_contiguous():
            return MiniArray(self.buffer, shape, c_strides(shape), self.offset)
        return MiniArray(self.tolist_flat(), shape)

    # -- materializing -----------------------------------------------------------------------
    def tolist_flat(self):
        return [self.item(*idx) for idx in product(*(range(n) for n in self.shape))]

    def tolist(self):
        return _nest(self.tolist_flat(), self.shape)


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _nest(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    chunk = _prod(shape[1:])
    return [_nest(flat[i * chunk:(i + 1) * chunk], shape[1:]) for i in range(shape[0])]


# -- broadcasting --------------------------------------------------------------------------------
def broadcast_shape(*shapes):
    """NumPy's rule: align shapes on the right; each pair of sizes must be equal or contain a 1."""
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out = []
    for sizes in zip(*padded):
        others = {n for n in sizes if n != 1}
        if len(others) > 1:
            raise ValueError(f"shapes {shapes} are not broadcastable")
        out.append(others.pop() if others else 1)
    return tuple(out)


def broadcast_to(a, shape):
    """A view where every stretched axis has stride 0: the same element is read again and again."""
    pad = len(shape) - len(a.shape)
    strides = [0] * pad
    for n_old, s_old, n_new in zip(a.shape, a.strides, shape[pad:]):
        if n_old == n_new:
            strides.append(s_old)
        elif n_old == 1:
            strides.append(0)
        else:
            raise ValueError(f"cannot broadcast {a.shape} to {shape}")
    return MiniArray(a.buffer, shape, strides, a.offset)


def add(a, b):
    shape = broadcast_shape(a.shape, b.shape)
    a2, b2 = broadcast_to(a, shape), broadcast_to(b, shape)
    out = [a2.item(*i) + b2.item(*i) for i in product(*(range(n) for n in shape))]
    return MiniArray(out, shape)


# -- einsum as named loops -----------------------------------------------------------------------
def mini_einsum(spec, x, y):
    """Two-operand einsum: every letter is a loop; letters missing from the output are summed."""
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
        result[key] = result.get(key, 0) + prod_
    shape = tuple(sizes[l] for l in out)
    flat = [result.get(idx, 0) for idx in product(*(range(n) for n in shape))]
    return MiniArray(flat, shape)
