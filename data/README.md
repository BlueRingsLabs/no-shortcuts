# Data

Small datasets that ship with the repo so the labs run offline. Everything else in the course is either
bundled with scikit-learn (`load_digits`, `load_breast_cancer`, ...) or generated synthetically inside the lab,
which I prefer anyway: when you generate the data yourself, you know the ground truth, and you can check
whether your model found it.

| File | Rows | What it is | Used in |
|---|---|---|---|
| `fuel_consumption_co2.csv` | 1,067 | Model-year 2014 light-duty vehicles in Canada: engine size, cylinders, fuel consumption and CO2 emissions. Derived from the public Natural Resources Canada fuel consumption ratings. | Linear regression, EDA |
| `telecust_1000.csv` | 1,000 | Telecom customers with demographics and a 4-class customer category (`custcat`). A classic teaching dataset for k-NN. | k-NN, classification |

If you add a dataset: keep it under a few MB, document its origin and license here, and never, ever commit
real personal data. I don't care how "anonymized" you think it is. Read lesson 15.2 first.
