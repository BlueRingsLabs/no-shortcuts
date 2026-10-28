import numpy as np
from mini_numpy import MiniArray, broadcast_to, add, broadcast_shape

print("--- demo 1: the four fields")
a = np.arange(12, dtype=np.int64).reshape(3, 4)
print(a.shape, a.dtype, a.strides)

print("--- demo 2: mini")
m = MiniArray.arange(12).reshape(3, 4)
print(m.shape, m.strides, m.offset)
print(m.item(1, 2), a[1, 2])

print("--- demo 3: slicing is a view")
mv = m[:, 1:3]
print(mv.shape, mv.strides, mv.offset, mv.tolist())
av = a[:, 1:3]
print(av.strides, np.shares_memory(a, av))
av[0, 0] = 100
print(a[0])
a[0, 1] = 1

print("--- demo 4: transpose")
print(m.T.shape, m.T.strides, a.T.strides, a.T.flags['C_CONTIGUOUS'])

print("--- demo 5: reshape of a transpose copies")
r1 = a.reshape(2, 6); r2 = a.T.reshape(12)
print(np.shares_memory(a, r1), np.shares_memory(a, r2))

print("--- demo 6: fancy and boolean indexing copy")
f = a[[0, 2]]
print(np.shares_memory(a, f))
b = np.arange(6)
b[b > 2] = 0
print(b)
c = np.arange(6)
c[c > 2][0] = 99
print(c)

print("--- demo 7: broadcasting")
print(broadcast_shape((3, 1), (4,)), np.broadcast_shapes((3, 1), (4,)))
col = MiniArray([10, 20, 30], (3, 1)); row = MiniArray.arange(4)
print(add(col, row).tolist())
print(np.array([[10], [20], [30]]) + np.arange(4))
bt = np.broadcast_to(np.arange(4), (3, 4))
print(bt.strides, bt.flags.writeable)
try:
    np.ones((3, 2)) + np.ones(3)
except ValueError as e:
    print("ValueError:", e)

print("--- demo 8: the classic mistake")
X = np.random.default_rng(0).normal(size=(5, 3))
mu = X.mean(axis=1)
try:
    X - mu
except ValueError as e:
    print("ValueError:", e)
print((X - X.mean(axis=1, keepdims=True)).mean(axis=1).round(12))
