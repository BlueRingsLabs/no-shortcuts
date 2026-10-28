import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([1.0, 2.0, 3.0, 4.0])
score = np.array([52.0, 61.0, 68.0, 80.0])

model = LinearRegression()
model.fit(hours, score)
