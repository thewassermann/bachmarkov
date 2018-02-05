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

	if sys.argv[1] == '1':

		chorales = data_utils.load_clean_chorales()

		# call only if not in memory
		SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

		major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

		cd_mh = {
				'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
				'NPI' : mh_boolean.NoParallelIntervals('NPI'),
				'CM' : mh_boolean.ContraryMotion('CM'),
				'NTT' : mh_boolean.NoteToTonic('NTT'),
				'LTS' : mh_boolean.LeapThenStep('LTS'),
				'RR' : mh_boolean.ReduceRepeated('RR'),
				'MWT' : mh_boolean.MovementWithinThird('MWT'),
			} 

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[10000, 1000],
			[10000, 1000, 100, 10, 1, .1, .01],
			True
		)

		pickle.dump(res, open( "RegularizationCV-10000-1000.p", "wb" ) )

	if sys.argv[1] == '2':

		chorales = data_utils.load_clean_chorales()

		# call only if not in memory
		SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

		major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

		cd_mh = {
				'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
				'NPI' : mh_boolean.NoParallelIntervals('NPI'),
				'CM' : mh_boolean.ContraryMotion('CM'),
				'NTT' : mh_boolean.NoteToTonic('NTT'),
				'LTS' : mh_boolean.LeapThenStep('LTS'),
				'RR' : mh_boolean.ReduceRepeated('RR'),
				'MWT' : mh_boolean.MovementWithinThird('MWT'),
			} 

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[100, 10],
			[10000, 1000, 100, 10, 1, .1, .01],
			True
		)

		pickle.dump(res, open( "RegularizationCV-100-10.p", "wb" ) )

	if sys.argv[1] == '3':

		chorales = data_utils.load_clean_chorales()

		# call only if not in memory
		SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

		major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

		cd_mh = {
				'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
				'NPI' : mh_boolean.NoParallelIntervals('NPI'),
				'CM' : mh_boolean.ContraryMotion('CM'),
				'NTT' : mh_boolean.NoteToTonic('NTT'),
				'LTS' : mh_boolean.LeapThenStep('LTS'),
				'RR' : mh_boolean.ReduceRepeated('RR'),
				'MWT' : mh_boolean.MovementWithinThird('MWT'),
			} 

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[1, .1],
			[10000, 1000, 100, 10, 1, .1, .01],
			True
		)

		pickle.dump(res, open( "RegularizationCV-1-01.p", "wb" ) )

	if sys.argv[1] == '4':

		chorales = data_utils.load_clean_chorales()

		# call only if not in memory
		SPLITDICT = pickle.load(open( "train_test_splits.p", "rb" ))

		major_train_chorales = [chorales['Major'][idx] for idx in SPLITDICT['major_train_idx']]

		cd_mh = {
				'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
				'NPI' : mh_boolean.NoParallelIntervals('NPI'),
				'CM' : mh_boolean.ContraryMotion('CM'),
				'NTT' : mh_boolean.NoteToTonic('NTT'),
				'LTS' : mh_boolean.LeapThenStep('LTS'),
				'RR' : mh_boolean.ReduceRepeated('RR'),
				'MWT' : mh_boolean.MovementWithinThird('MWT'),
			} 

		res = regularization.MHCV(
			major_train_chorales,
			10,
			cd_mh,
			[.01],
			[10000, 1000, 100, 10, 1, .1, .01],
			True
		)

		pickle.dump(res, open( "RegularizationCV-001.p", "wb" ) )

if __name__ == '__main__':
   main()
