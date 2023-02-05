import timeit

from sklearn.datasets import fetch_openml

from train import train


def do_predict(pipeline, X):
    return pipeline.predict_proba(X[['pclass', 'sex', 'age', 'fare']])


def main():
    pipeline = train()
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser='liac-arff')

    def evaluate_predict():
        do_predict(pipeline, X)

    t = timeit.Timer(lambda: evaluate_predict())
    print(t.repeat(repeat=3, number=1))


if __name__ == '__main__':
    main()
