from sklearn.datasets import load_diabetes
import pandas as pd

dataset = load_diabetes() # I would use as_frame=True, but I think the Docker env doesn't have the right update for it.
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(df.head())
xs = df['bmi']
ys = df['bp']


