from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def train():
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True, parser='liac-arff')
    X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)

    pipeline = Pipeline([
        ('encode_sex',
         ColumnTransformer(
             [
                 ("age", MissingIndicator(), ["age"]),
                 ("fare", MissingIndicator(), ["fare"]),
                 ("sex", OrdinalEncoder(), ["sex"]),
                 ("pclass", OrdinalEncoder(), ["pclass"]),
             ],
             remainder="passthrough"
         )
         ),
        ('model', RandomForestClassifier()),
    ])
    pipeline.fit(X[['pclass', 'sex', 'age', 'fare']], y)
    return pipeline
