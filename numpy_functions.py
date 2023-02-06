import timeit

import numpy as np
import pandas as pd


def next_seq():
    return np.ones(1000)


def py_benchmark():
    t = timeit.Timer(lambda: max(next_seq()))
    out = t.repeat(repeat=100, number=1)
    print((pd.DataFrame({'t, milliseconds': out}) * 1000).describe())
