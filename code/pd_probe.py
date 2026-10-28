import warnings
import numpy as np, pandas as pd
print(pd.__version__, pd.options.mode.copy_on_write if hasattr(pd.options.mode, "copy_on_write") else "no option")
s1 = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
s2 = pd.Series([10.0, 20.0, 30.0], index=["b", "c", "d"])
print(s1 + s2)
print(s1.values + s2.values)
print(s1.add(s2, fill_value=0))
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
print(df.dtypes)
print(type(df["temp"].to_numpy()), df["city"].dtype, type(df["city"].dtype))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df[df["temp"] > 10]["temp"] = 0
    print([type(x.message).__name__ + ": " + str(x.message)[:160] for x in w])
print(df)
df.loc[df["temp"] > 10, "temp"] = 0
print(df)
df = pd.DataFrame({"city": ["Rome", "Lima", "Rome", "Oslo"], "temp": [20, 18, 24, 5]})
sub = df[df["temp"] > 10]
sub["temp"] = 0
print(df["temp"].tolist(), sub["temp"].tolist())
col = df["temp"]
col.iloc[0] = 99
print(df["temp"].tolist())
print(df.groupby("city")["temp"].mean())
print(df.groupby("city", sort=False)["temp"].mean())
s = pd.Series([5, 6, 7], index=[2, 1, 0])
print(s[0], s.iloc[0])
