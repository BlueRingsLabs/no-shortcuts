"""Check MiniArray against real NumPy on random shapes, slices, transposes, broadcasts and einsums."""
import random

import numpy as np

from mini_numpy import MiniArray, add, broadcast_shape, broadcast_to, mini_einsum

rng = random.Random(0)


def to_np(m):
    return np.array(m.tolist())


def random_slice(n):
    start, stop = sorted(rng.randrange(-n, n + 1) for _ in range(2))
    step = rng.choice([1, 1, 2, 3, -1, -2])
    return slice(start, stop, step) if step > 0 else slice(stop, start, step)


checks = 0
for _ in range(500):
    shape = tuple(rng.randint(1, 5) for _ in range(rng.randint(1, 3)))
    size = int(np.prod(shape))
    m = MiniArray.arange(size).reshape(*shape)
    a = np.arange(size).reshape(shape)

    # strides: ours are in elements, NumPy's in bytes
    assert tuple(s * a.itemsize for s in m.strides) == a.strides

    key = tuple(rng.randrange(n) if rng.random() < 0.3 else random_slice(n) for n in shape)
    mv, av = m[key], a[key]
    assert mv.shape == av.shape, (shape, key, mv.shape, av.shape)
    if av.size:
        assert np.array_equal(to_np(mv), av), (shape, key)
        assert tuple(s * a.itemsize for s in mv.strides) == av.strides
    assert np.array_equal(to_np(m.T), a.T)
    checks += 1

for _ in range(300):
    ndim = rng.randint(1, 3)
    s1 = tuple(rng.choice([1, 2, 3]) for _ in range(ndim))
    s2 = tuple(rng.choice([1, n]) for n in s1)[rng.randint(0, ndim - 1):]
    try:
        expected = np.broadcast_shapes(s1, s2)
    except ValueError:
        expected = None
    try:
        got = broadcast_shape(s1, s2)
    except ValueError:
        got = None
    assert got == expected, (s1, s2, got, expected)
    if expected is not None:
        x = MiniArray.arange(int(np.prod(s1))).reshape(*s1)
        y = MiniArray.arange(int(np.prod(s2))).reshape(*s2)
        assert np.array_equal(to_np(add(x, y)), np.arange(int(np.prod(s1))).reshape(s1) + np.arange(int(np.prod(s2))).reshape(s2))
    checks += 1

# broadcast_to really uses stride 0, like NumPy
b = broadcast_to(MiniArray.arange(4), (3, 4))
assert b.strides == (0, 1) and np.broadcast_to(np.arange(4), (3, 4)).strides == (0, 8)

for spec, s1, s2 in [("ij,jk->ik", (2, 3), (3, 4)), ("ij,ij->i", (3, 4), (3, 4)), ("i,j->ij", (3,), (4,)),
                     ("bij,bjk->bik", (2, 2, 3), (2, 3, 2)), ("ij,jk->", (2, 3), (3, 2))]:
    x = MiniArray.arange(int(np.prod(s1))).reshape(*s1)
    y = MiniArray.arange(int(np.prod(s2))).reshape(*s2)
    want = np.einsum(spec, np.arange(int(np.prod(s1))).reshape(s1), np.arange(int(np.prod(s2))).reshape(s2))
    got = mini_einsum(spec, x, y)
    assert np.array_equal(np.array(got.tolist()), want), spec
    checks += 1

print(f"{checks} checks against NumPy {np.__version__}: all passed")
