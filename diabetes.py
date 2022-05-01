from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

dataset = load_diabetes() # I would use as_frame=True, but I think the Docker env doesn't have the right update for it.
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(df.head())
xs = df['bmi']
xs_reshape = np.array(xs).reshape(-1 ,1)
ys = df['bp']

model = LinearRegression()
model.fit(xs_reshape, ys)
print(f'Model: y = {model.coef_[0]}x + {model.intercept_}')


