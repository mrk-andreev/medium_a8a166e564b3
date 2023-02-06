import timeit

import pandas as pd
from sklearn.datasets import fetch_openml

from train import train


def load_dataset():
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser='liac-arff')
    return X


def do_predict(pipeline, X):
    return pipeline.predict_proba(X[['pclass', 'sex', 'age', 'fare']])[:, 0].astype(float).tolist()


def py_benchmark():
    pipeline = train()
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser='liac-arff')

    def evaluate_predict():
        do_predict(pipeline, X)

    t = timeit.Timer(lambda: evaluate_predict())
    out = t.repeat(repeat=100, number=1)
    print((pd.DataFrame({'t, milliseconds': out}) * 1000).describe())
