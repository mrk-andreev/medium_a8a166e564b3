# Compare speed of interaction

## ML Pipeline

For `pipeline` fitted on `Titanic`: 

```
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
```

We compare python call (look at [predict_pipeline.py](predict_pipeline.py)):

```
def do_predict(pipeline, X):
    return pipeline.predict_proba(X[['pclass', 'sex', 'age', 'fare']])
```

with R -> Python call [predict_pipeline.R](predict_pipeline.R):

```
library("microbenchmark")
microbench_out <- microbenchmark(do_predict(pipeline, X))
microbench_out
```


### Result

#### R

```
Unit: milliseconds
                    expr      min       lq     mean   median       uq      max
 do_predict(pipeline, X) 55.42288 66.04502 85.75898 81.54774 98.76258 137.4046
 neval
   100
```

#### Python

```
       t, milliseconds
mean         50.587171
std          15.673094
min          30.254079
25%          39.684958
50%          45.808600
75%          58.351191
max         118.226873
```

## NumPy 

```
def next_seq():
    return np.ones(1000)
    
max(next_seq())
```

#### R

```
Unit: microseconds
            expr     min      lq     mean   median      uq     max neval
 max(next_seq()) 291.006 295.749 309.0466 298.3855 305.188 634.687   100
```

#### Python

```
       t, milliseconds
mean          0.259414
std           0.096316
min           0.144433
25%           0.221229
50%           0.225632
75%           0.248448
max           0.515984
```
