library(reticulate)
use_python("/usr/bin/python3")

source_python("numpy_functions.py")

library("microbenchmark")
microbench_out <- microbenchmark(max(next_seq()))
microbench_out
