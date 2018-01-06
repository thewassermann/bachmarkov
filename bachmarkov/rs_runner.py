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

import sys

def main():
	"""
	Command line interface for  random search:

	Parameters
	----------

		n_iter : int
		run_length : int
		walkers : int
		out_filename :str
	"""

	if len(sys.argv) != 5:
		print('Input correct number of Parameters please')
		return -1

	n_iter = int(sys.argv[1])
	run_length = int(sys.argv[2])
	walkers = int(sys.argv[3])
	out_filename = sys.argv[4] + ".p"

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
		n_iter=n_iter,
		run_length=run_length,
		walkers=walkers
	)

	pickle.dump(RS_output, open(out_filename, "wb" ))


if __name__ == '__main__':
   main()
