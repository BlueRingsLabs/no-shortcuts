import inspect
import numpy as np
from sklearn.preprocessing import StandardScaler
print(inspect.signature(np.reshape))
print(inspect.signature(StandardScaler))
print(StandardScaler.__module__, inspect.getfile(StandardScaler).split("site-packages/")[1])
print([c.__name__ for c in StandardScaler.__mro__])
print([n for n in dir(StandardScaler()) if not n.startswith("_")])
src = inspect.getsource(StandardScaler.fit)
print(len(src.splitlines()), "lines in StandardScaler.fit")
print(src[:900])
