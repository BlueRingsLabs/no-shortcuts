# %% [markdown]
# # Lab 48.2: Data versioning, pipelines and environments
#
# 1. Content-addressed data versioning (the idea under DVC and git-lfs): the repository stores small pointer files;
#    the data lives in a cache keyed by hash. Switch versions, and see that identical data is stored once.
# 2. A pipeline with stage caching: each stage declares its inputs, parameters and outputs, and reruns only when one
#    of them changed (DVC pipelines, Make, most orchestrators).
# 3. Reproducing a result: same inputs, same code, same environment. What's bit-identical and what isn't.

# %%
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

WORK = Path(tempfile.mkdtemp(prefix="dataver_"))
CACHE = WORK / ".cache"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# %% [markdown]
# ## 1. Content-addressed data versioning

# %%
def add(path):
    """Move a data file into the cache under its hash; leave a pointer file (the thing you commit to git)."""
    sha = sha256_file(path)
    target = CACHE / sha[:2] / sha
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    pointer = path.with_suffix(path.suffix + ".ptr")
    pointer.write_text(json.dumps({"sha256": sha, "size": path.stat().st_size, "path": path.name}))
    return pointer


def checkout(pointer):
    meta = json.loads(pointer.read_text())
    dest = pointer.parent / meta["path"]
    shutil.copy2(CACHE / meta["sha256"][:2] / meta["sha256"], dest)
    assert sha256_file(dest) == meta["sha256"], "cache corrupted"
    return dest


data = WORK / "transactions.csv"
rs = np.random.default_rng(0)
rows = rs.normal(size=(20_000, 6))
np.savetxt(data, rows, delimiter=",", fmt="%.6f")
ptr_v1 = add(data)
v1_pointer = ptr_v1.read_text()
np.savetxt(data, np.vstack([rows, rs.normal(size=(2_000, 6))]), delimiter=",", fmt="%.6f")   # new rows arrive
add(data)
v2_pointer = ptr_v1.read_text()
np.savetxt(data, rows, delimiter=",", fmt="%.6f")                                   # someone regenerates v1 exactly
add(data)
print(f"pointer file for v1: {v1_pointer}")
print(f"cache holds {sum(1 for _ in CACHE.rglob('*') if _.is_file())} objects after three adds (v1, v2, v1 again)")
ptr_v1.write_text(v1_pointer)
restored = checkout(ptr_v1)
print(f"checked out v1 again: {sum(1 for _ in open(restored))} rows, hash verified")
print("git versions the pointers (tiny, diffable); the cache (local, or a bucket) holds each distinct content once. A")
print("commit then pins code *and* data: check out a commit and you get the exact bytes the model was trained on.")
assert sum(1 for _ in CACHE.rglob("*") if _.is_file()) == 2

# %% [markdown]
# ## 2. A pipeline with stage caching

# %%
STATE = WORK / "pipeline.lock"
calls = []


def stage(name, fn, deps, params, outs):
    """Run fn only if the hash of (code, params, input contents) changed since the last successful run."""
    lock = json.loads(STATE.read_text()) if STATE.exists() else {}
    key = hashlib.sha256(json.dumps({"code": fn.__code__.co_code.hex(), "params": params,
                                     "deps": [sha256_file(d) for d in deps]}, sort_keys=True).encode()).hexdigest()
    if lock.get(name, {}).get("key") == key and all(Path(o).exists() and sha256_file(o) == lock[name]["outs"][str(o)] for o in outs):
        return "cached"
    fn(*deps, *outs, **params)
    calls.append(name)
    lock[name] = {"key": key, "outs": {str(o): sha256_file(o) for o in outs}}
    STATE.write_text(json.dumps(lock, indent=1))
    return "ran"


def prepare(src, dst, clip):
    x = np.loadtxt(src, delimiter=",")
    np.save(dst, np.clip(x, -clip, clip))


def featurize(src, dst):
    x = np.load(src)
    np.save(dst, np.column_stack([x, x[:, :2] ** 2]))


def train(src, dst, l2):
    x = np.load(src)
    y = (x[:, 0] + x[:, 1] ** 2 > 1).astype(float)
    A = x[:, 1:]
    w = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ y)
    np.save(dst, w)


P, F_, M = WORK / "prepared.npy", WORK / "features.npy", WORK / "model.npy"
params = {"prepare": {"clip": 3.0}, "train": {"l2": 1.0}}


def run_pipeline():
    return [stage("prepare", prepare, [data], params["prepare"], [P]),
            stage("featurize", featurize, [P], {}, [F_]),
            stage("train", train, [F_], params["train"], [M])]


print("\nfirst run:                      ", run_pipeline())
print("again, nothing changed:         ", run_pipeline())
params["train"]["l2"] = 10.0
print("changed a training parameter:   ", run_pipeline())
np.savetxt(data, rows[:, ::-1], delimiter=",", fmt="%.6f")                          # the raw data changed
print("changed the raw data:           ", run_pipeline())
params["prepare"]["clip"] = 3.0                                                     # same value as before: no change
print("'changed' a parameter to the same value:", run_pipeline())
print("each stage reruns only when its code, parameters or inputs change, and everything downstream of a change reruns.")
print("That's what makes a 3-hour pipeline cheap to iterate on, and what makes 'which data trained this?' answerable.")
assert calls.count("prepare") == 2 and calls.count("train") == 3

# %% [markdown]
# ## 3. Reproducing a result, bit for bit?

# %%
def train_torch(seed, threads):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    X = torch.randn(4096, 256)
    y = (X[:, :5].sum(1) > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 1))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(50):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], y)
        opt.zero_grad(); loss.backward(); opt.step()
    w = torch.cat([p.detach().flatten() for p in net.parameters()])
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16], loss.item()


a = train_torch(0, 4)
b = train_torch(0, 4)
c = train_torch(1, 4)
d = train_torch(0, 1)
torch.set_num_threads(4)
print(f"\nsame seed, same threads:       weights {a[0]}  loss {a[1]:.8f}")
print(f"same seed, same threads again: weights {b[0]}  loss {b[1]:.8f}   identical: {a[0] == b[0]}")
print(f"different seed:                weights {c[0]}  loss {c[1]:.8f}")
print(f"same seed, 1 thread:           weights {d[0]}  loss {d[1]:.8f}   identical to 4 threads: {a[0] == d[0]}")
print("floating-point addition isn't associative, so anything that changes the order of a sum (thread count, a")
print("different BLAS, a GPU kernel choice, batch composition) changes the last bits, and training amplifies them.")
print("Aim for bit-identical reproduction on the same hardware and software (seeds, pinned versions, deterministic")
print("algorithms), and statistical reproduction everywhere else: the same metrics within seed variance (48.1).")
assert a == b and a[0] != c[0]

print("\nAll checks passed.")
