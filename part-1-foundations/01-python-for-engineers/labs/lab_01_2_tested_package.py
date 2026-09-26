# %% [markdown]
# # Lab 01.2: A tiny package, tested properly
#
# This lab writes a real package to a temporary directory (src layout, pyproject.toml, tests), then runs pytest
# on it as a subprocess, exactly like CI would. Two versions of the package are tested:
#
# - `buggy`: a `standardize` that divides by zero on constant input and a `precision` with the wrong FP definition.
# - `fixed`: the corrected versions.
#
# The lab asserts that the test suite FAILS on the buggy version and PASSES on the fixed one. A test suite that
# can't tell those two apart is decoration.

# %%
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# %%
PYPROJECT = """
[project]
name = "minstats"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -p no:cacheprovider"
"""

BUGGY = '''
from collections.abc import Sequence


def standardize(xs: Sequence[float]) -> list[float]:
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return [(x - m) / sd for x in xs]


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)   # wrong: these are false negatives
    return tp / (tp + fp)
'''

FIXED = '''
from collections.abc import Sequence


def standardize(xs: Sequence[float], rel_eps: float = 1e-9) -> list[float]:
    """Zero mean, unit variance. Constant input returns zeros instead of dividing by zero:
    a constant feature carries no information, and zeros say exactly that.

    "Constant" is judged RELATIVE to the magnitude of the data. For [699051.258, 699051.258, 699051.258]
    the computed mean is off by one ulp, so the computed sd is ~1e-10 instead of 0. An absolute check
    like `sd < 1e-12` misses that and divides rounding noise by rounding noise (hypothesis found this)."""
    if len(xs) == 0:
        raise ValueError("standardize of empty sequence")
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    if sd <= rel_eps * max(1.0, abs(m)):
        return [0.0 for _ in xs]
    return [(x - m) / sd for x in xs]


def precision(y_true: Sequence[int], y_pred: Sequence[int], zero_division: float = 0.0) -> float:
    """TP / (TP + FP). If nothing was predicted positive, return `zero_division`."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return tp / (tp + fp) if (tp + fp) else zero_division
'''

TESTS = '''
import math

import pytest
from hypothesis import given, settings, strategies as st

from minstats.core import precision, standardize

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.lists(finite, min_size=2, max_size=50))
def test_standardize_zero_mean(xs):
    z = standardize(xs)
    assert len(z) == len(xs)
    assert abs(sum(z) / len(z)) < 1e-6


def test_standardize_known_values():
    assert standardize([1.0, 3.0]) == pytest.approx([-1.0, 1.0])


@pytest.mark.parametrize("y_true, y_pred, expected", [
    ([0, 1], [1, 1], 0.5),      # TP=1, FP=1
    ([1, 1], [1, 0], 1.0),      # TP=1, FP=0 (the FN must not count)
    ([1, 0, 1, 0], [1, 1, 1, 0], 2 / 3),
])
def test_precision_hand_computed(y_true, y_pred, expected):
    assert math.isclose(precision(y_true, y_pred), expected)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.lists(st.tuples(st.integers(0, 1), st.integers(0, 1)), min_size=1, max_size=40))
def test_precision_matches_reference(pairs):
    """Reference-implementation test: compare against a definition written independently."""
    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    predicted_pos = [t for t, p in pairs if p == 1]
    if not predicted_pos:
        return                                   # undefined case, covered by its own decision in the code
    expected = sum(predicted_pos) / len(predicted_pos)
    assert math.isclose(precision(y_true, y_pred), expected)
'''


def build_package(root: Path, core_source: str) -> None:
    (root / "src" / "minstats").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(textwrap.dedent(PYPROJECT))
    (root / "src" / "minstats" / "__init__.py").write_text("")
    (root / "src" / "minstats" / "core.py").write_text(textwrap.dedent(core_source))
    (root / "tests" / "test_core.py").write_text(textwrap.dedent(TESTS))


def run_pytest(root: Path) -> subprocess.CompletedProcess:
    # In a real project you'd `pip install -e .`; here we put src/ on the path to keep the lab fast and offline.
    env = {"PYTHONPATH": str(root / "src"), "PATH": "", "HYPOTHESIS_STORAGE_DIRECTORY": str(root / ".hyp")}
    return subprocess.run([sys.executable, "-m", "pytest"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)

# %% [markdown]
# ## Run the suite against both versions

# %%
results = {}
for name, source in [("buggy", BUGGY), ("fixed", FIXED)]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_package(root, source)
        proc = run_pytest(root)
        results[name] = proc.returncode
        last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        print(f"{name:6s} -> exit code {proc.returncode}: {last}")
        if name == "buggy":
            # Show that hypothesis found the constant-input case on its own.
            falsifying = [l for l in proc.stdout.splitlines() if "Falsifying example" in l or "xs=" in l]
            print("   hypothesis says:", *falsifying[:3], sep="\n     ")

# %%
assert results["buggy"] != 0, "the test suite should catch the bugs"
assert results["fixed"] == 0, "the fixed package should pass every test"
print("\nThe suite distinguishes buggy from fixed code. That's the whole point of a test suite.")

# %% [markdown]
# ## The bug I didn't plan
#
# The first version of FIXED in this lab used an absolute threshold, `if sd < 1e-12`. Hypothesis broke it
# within a second with `xs=[699051.2584180862] * 3`: the mean of three identical floats is not exactly that float
# (rounding in `sum`), so the standard deviation came out around 1e-10, passed the check, and every output was
# -1.0 instead of 0.0. I left the story here because it's the best argument for property-based testing I have:
# it found a bug in code written by someone who was, at that very moment, writing a lesson about the bug's cousin.
# Module 07.2 explains the floating point side properly.
#
# ## Try this
#
# - Put the absolute threshold back (`if sd < 1e-12:`) and rerun. Watch hypothesis shrink the failing input.
#
# - Delete `test_precision_hand_computed`. Does the suite still catch the precision bug? (It should: the
#   reference test catches it too. Two independent tests for one bug is good.)
# - In FIXED, change the constant-input behaviour to `raise ValueError`. Which test fails, and how would you
#   change the test to encode that decision instead?
