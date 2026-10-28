import numpy as np, sys
print(np.__version__, sys.version.split()[0])
a = np.arange(12, dtype=np.int64).reshape(3, 4)
print(a.shape, a.strides, a.dtype, a.itemsize, a.nbytes, a.flags['C_CONTIGUOUS'])
b = a[:, 1:3]
print(b.shape, b.strides, np.shares_memory(a, b), b.base is a, b.flags['C_CONTIGUOUS'])
t = a.T
print(t.shape, t.strides, np.shares_memory(a,t), t.flags['C_CONTIGUOUS'], t.flags['F_CONTIGUOUS'])
f = a[[0, 2]]
print(np.shares_memory(a, f), f.base is None)
r = t.reshape(12)
print(np.shares_memory(t, r))
r2 = a.reshape(2,6); print(np.shares_memory(a, r2))
try:
    np.array(t, copy=False)
    print("copy=False ok")
except ValueError as e:
    print("copy=False error:", e)
try:
    x = t.view(); x.shape = (12,)
except Exception as e:
    print(type(e).__name__, e)
bt = np.broadcast_to(np.arange(4), (3,4)); print(bt.strides, bt.flags.writeable)
print((np.ones((3,1)) + np.arange(4)).shape)
try:
    np.ones((3,2)) + np.ones(3)
except ValueError as e: print(e)
print(np.einsum('ij,jk->ik', np.ones((2,3)), np.ones((3,4))).shape)
