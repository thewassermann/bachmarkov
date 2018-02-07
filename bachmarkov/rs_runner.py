from  music21 import *
import numpy as np
import pandas as pd

import matplotlib 
matplotlib.use('Agg')

from mh import mh, mh_boolean, mowile
from gibbs import gibbs, gibbs_boolean
from tuning import convergence_diagnostics
from tuning import random_search
from tuning import ll_method
from tuning import regularization 

from utils import chord_utils, extract_utils, data_utils

import seaborn as sns

import pickle

import sys


def main():
	"""
	Command line interface for  random search:

	Parameters
	----------

		num : which function to run:
	"""

	cd_gibbs = {
			'NC' : gibbs_boolean.NoCrossing('NC'),
			'SWM' : gibbs_boolean.StepwiseMotion('SWM'),
			'NPI' : gibbs_boolean.NoParallelIntervals('NPI'),
			'OM' : gibbs_boolean.OctaveMax('OM'),
			'NN' : gibbs_boolean.NewNotes('NN'), 
		}

	chorales = data_utils.load_clean_chorales()

	# call only if not in memory
	SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

	major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

	if sys.argv[1] == '1':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[100000],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-100000.p", "wb" ) )

	if sys.argv[1] == '2':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[10000],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-10000.p", "wb" ) )

	if sys.argv[1] == '3':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[1000],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-1000.p", "wb" ) )


	if sys.argv[1] == '4':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[100],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-100.p", "wb" ) )


	if sys.argv[1] == '5':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[10],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-10.p", "wb" ) )

	if sys.argv[1] == '6':

		res = regularization.MHCVGibbs(
			major_train_chorales,
			10,
			cd_gibbs,
			[1],
			[100000, 10000, 1000, 100, 10, 1],
		)

		pickle.dump(res, open( "GibbsRegularizationCV-1.p", "wb" ) )

if __name__ == '__main__':
   main()
