import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


def get_model(xs_reshape, ys):
    model = LinearRegression()
    model.fit(xs_reshape, ys)
    print(f'Model: y = {model.coef_[0]}x + {model.intercept_}')
    return model

def plot_model_and_data(xs, ys, model, xs_reshape):
    plt.figure()
    plt.title("Diabetes")
    plt.xlabel("BMI")
    plt.ylabel("Blood pressure")
    plt.scatter(xs, ys, s=20)
    plt.plot(xs, model.predict(xs_reshape), linewidth=3, c='k')
    plt.show()

def predict_for_x(model, x):
    print(f'BP for BMI={x}: {model.predict([[x]])[0]}')

if __name__ == '__main__':
    dataset = load_diabetes() # I would use as_frame=True, but I think the Docker env doesn't have the right update for it.
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # print(df.head())
    xs = df['bmi']
    xs_reshape = np.array(xs).reshape(-1 ,1)
    ys = df['bp']

    model = get_model(xs_reshape, ys)
    plot_model_and_data(xs, ys, model, xs_reshape)
    predict_for_x(model, 2)
