"""Check the pandas miniature against pandas on random data."""
import random

import numpy as np
import pandas as pd

from mini_pandas import MiniDataFrame, MiniSeries

rng = random.Random(0)
LABELS = list("abcdefgh")


def to_pd(m):
    return pd.Series(m.values, index=m.index, name=m.name, dtype="float64")


checks = 0

# 1. alignment: random overlapping indexes, with and without fill_value
for _ in range(400):
    i1 = rng.sample(LABELS, rng.randint(1, 6))
    i2 = i1 if rng.random() < 0.2 else rng.sample(LABELS, rng.randint(1, 6))
    v1 = [float(rng.randint(-9, 9)) for _ in i1]
    v2 = [float(rng.randint(-9, 9)) for _ in i2]
    p1, p2 = pd.Series(v1, index=i1), pd.Series(v2, index=i2)
    m1, m2 = MiniSeries(v1, i1), MiniSeries(v2, i2)
    pd.testing.assert_series_equal(to_pd(m1 + m2), p1 + p2, check_index_type=False)
    pd.testing.assert_series_equal(to_pd(m1.add(m2, fill_value=0.0)), p1.add(p2, fill_value=0.0),
                                   check_index_type=False)
    checks += 2

# 2. label vs position on an integer index that isn't 0..n-1 in order
for _ in range(200):
    n = rng.randint(1, 6)
    idx = rng.sample(range(10), n)
    vals = [rng.randint(0, 99) for _ in range(n)]
    p, m = pd.Series(vals, index=idx), MiniSeries(vals, idx)
    label, pos = rng.choice(idx), rng.randrange(n)
    assert m[label] == p[label] and m.iloc(pos) == p.iloc[pos]
    checks += 1

# 3. groupby: random keys and values, mean and sum, sorted keys
for _ in range(300):
    n = rng.randint(1, 12)
    keys = [rng.choice(["Rome", "Lima", "Oslo", "Kyiv"]) for _ in range(n)]
    vals = [rng.randint(-20, 40) for _ in range(n)]
    pdf = pd.DataFrame({"city": keys, "temp": vals})
    mdf = MiniDataFrame({"city": keys, "temp": vals})
    got = mdf.groupby("city").mean("temp")
    want = pdf.groupby("city")["temp"].mean()
    pd.testing.assert_series_equal(to_pd(got), want, check_index_type=False, check_names=False)
    got = mdf.groupby("city").sum("temp")
    want = pdf.groupby("city")["temp"].sum().astype("float64")
    pd.testing.assert_series_equal(to_pd(got), want, check_index_type=False, check_names=False)
    checks += 2

# 4. copy-on-write: writing to a selected column never changes the frame, like pandas >= 3.0
for _ in range(100):
    vals = [rng.randint(0, 9) for _ in range(5)]
    pdf, mdf = pd.DataFrame({"x": vals}), MiniDataFrame({"x": vals})
    pcol, mcol = pdf["x"], mdf["x"]
    label = rng.randrange(5)
    pcol.loc[label] = 100
    mcol.set_loc(label, 100)
    assert pdf["x"].tolist() == mdf["x"].values == vals
    assert pcol.tolist() == mcol.values
    checks += 1

print(f"{checks} checks against pandas {pd.__version__}: all passed")
