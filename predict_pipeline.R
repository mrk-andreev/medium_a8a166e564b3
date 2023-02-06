library(reticulate)
use_python("/usr/bin/python3")
source_python("train.py")

pipeline <- train()
source_python("predict_pipeline.py")

X <- load_dataset()

library("microbenchmark")
microbench_out <- microbenchmark(do_predict(pipeline, X))
microbench_out

py_benchmark()
