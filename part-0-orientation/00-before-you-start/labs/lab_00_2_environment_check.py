# %% [markdown]
# # Lab 00.2: Is your environment sane?
#
# Run this from inside your virtual environment. It checks the things that, in my experience, cause 90% of
# "the lab doesn't work" messages: wrong Python, system Python, missing packages, a slow numpy build.
# Every check prints what it found and, if it fails, what to do about it.

# %%
import importlib
import os
import platform
import sys
import time

problems: list[str] = []


def check(ok: bool, msg_ok: str, msg_fix: str) -> None:
    print(("  ok    " if ok else "  FIX   ") + (msg_ok if ok else msg_fix))
    if not ok:
        problems.append(msg_fix)


# %% [markdown]
# ## 1. Python version and virtual environment
#
# `sys.prefix != sys.base_prefix` is how Python itself knows it's running inside a venv.

# %%
print("Python")
check(sys.version_info >= (3, 11), f"Python {platform.python_version()}",
      f"Python {platform.python_version()} is too old; install 3.11+ (uv python install 3.11)")
in_venv = sys.prefix != sys.base_prefix or "CONDA_PREFIX" in os.environ or os.environ.get("CI") == "true"
check(in_venv, f"running inside an environment: {sys.prefix}",
      "you're using the system Python; create and activate a venv (see lesson 00.2)")

# %% [markdown]
# ## 2. Core packages
#
# If any of these is missing, `pip install -r requirements.txt` from the repository root.

# %%
print("Packages")
core = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "torch", "yaml"]
for name in core:
    try:
        mod = importlib.import_module(name)
        check(True, f"{name} {getattr(mod, '__version__', '?')}", "")
    except ImportError:
        check(False, "", f"{name} is not installed")

# %% [markdown]
# ## 3. Is numpy fast?
#
# A 1000x1000 matrix multiplication is about 2 GFLOP (2 * n^3 floating point operations). Any numpy built on an
# optimized BLAS (OpenBLAS, MKL, Accelerate) does that in well under a second on a laptop. If it takes several
# seconds, your numpy is linked against a reference BLAS and everything in this course will be painfully slow.

# %%
import numpy as np

rng = np.random.default_rng(0)
a = rng.standard_normal((1000, 1000))
b = rng.standard_normal((1000, 1000))
a @ b  # warm-up: the first call pays for thread pool startup
t0 = time.perf_counter()
for _ in range(3):
    a @ b
dt = (time.perf_counter() - t0) / 3
gflops = 2 * 1000**3 / dt / 1e9
print("Speed")
check(dt < 1.0, f"1000x1000 matmul in {dt * 1000:.0f} ms (~{gflops:.0f} GFLOP/s)",
      f"matmul took {dt:.2f} s; reinstall numpy from wheels (pip install --force-reinstall numpy)")

# %% [markdown]
# ## 4. GPU (optional)
#
# Not having a GPU is fine. Having one that torch can't see is the thing to fix.

# %%
import torch

print("GPU")
if torch.cuda.is_available():
    print(f"  ok    CUDA device: {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("  ok    Apple MPS device available")
else:
    print("  info  no GPU visible to torch; CPU is fine until Part IV (see lesson 00.2 if you expected one)")

# %% [markdown]
# ## Verdict

# %%
if problems:
    print("\nThings to fix:")
    for p in problems:
        print(" -", p)
assert not problems, f"{len(problems)} problem(s) found, see above"
print("\nEnvironment looks sane. On to 00.3.")
