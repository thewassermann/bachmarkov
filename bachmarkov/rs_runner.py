from  music21 import *
import numpy as np
import pandas as pd

import matplotlib 
matplotlib.use('Agg')

from mh import mh_boolean
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

	cd_mh = {
		'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
		'NPI' : mh_boolean.NoParallelIntervals('NPI'),
		'CM' : mh_boolean.ContraryMotion('CM'),
		'NTT' : mh_boolean.NoteToTonic('NTT'),
		'LTS' : mh_boolean.LeapThenStep('LTS'),
		'SWM' : mh_boolean.StepwiseMotion('SWM'),
		'FD' : mh_boolean.FollowDirection('FD')
	}

	chorales = data_utils.load_clean_chorales()

	# call only if not in memory
	SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

	major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

	if sys.argv[1] == '1':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[100000],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-100000.p", "wb" ) )

	if sys.argv[1] == '2':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[10000],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-10000.p", "wb" ) )

	if sys.argv[1] == '3':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[1000],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-1000.p", "wb" ) )


	if sys.argv[1] == '4':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[100],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-100.p", "wb" ) )


	if sys.argv[1] == '5':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[10],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-10.p", "wb" ) )

	if sys.argv[1] == '6':

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[1],
			[100000, 10000, 1000, 100, 10, 1],
			True
		)

		pickle.dump(res, open( "RegularizationCV-1.p", "wb" ) )

if __name__ == '__main__':
   main()
