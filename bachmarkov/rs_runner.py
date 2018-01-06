from  music21 import *
import numpy as np
import pandas as pd

from mh import mh
from gibbs import gibbs
from tuning import convergence_diagnostics
from tuning import random_search

from utils import chord_utils, extract_utils, data_utils

import seaborn as sns

import pickle

chorales = data_utils.load_clean_chorales(n_upload=20)

mh_algo = mh.PachetRoySopranoAlgo(
    chorales['Major'][1],
)

RS = random_search.RandomSearch(
    np.random.choice(chorales['Major'], size=3),
    mh_algo,
    'MH'
)

RS_output = RS.run(
    n_iter=3,
    run_length=100,
    walkers=3
)

pickle.dump(RS_output, open( "RS_output.p", "wb" ) )