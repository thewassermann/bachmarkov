from  music21 import *
import numpy as np
import pandas as pd

from mh import mh, mh_boolean
from gibbs import gibbs, gibbs_boolean
from tuning import convergence_diagnostics
from tuning import random_search
from tuning import ll_method

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

	if len(sys.argv) != 2:
		print('Input correct number of Parameters please')
		return -1

	num = int(sys.argv[1])

	chorales = data_utils.load_clean_chorales(n_upload=30)

	# mh run
	if num == 1:
		cd_mh = mh_boolean.create_cross_constraint_dict(
		{
			'NIJ' : mh_boolean.NoIllegalJumps('NIJ'),
			'NPI' : mh_boolean.NoParallelIntervals('NPI'),
			'CM' : mh_boolean.ContraryMotion('CM'),
			'NTT' : mh_boolean.NoteToTonic('NTT'),
			'LTS' : mh_boolean.LeapThenStep('LTS'),
			'RR' : mh_boolean.ReduceRepeated('RR'),
			'MWT' : mh_boolean.MovementWithinThird('MWT'),
		},
		'MH'
	)

	test_ = ll_method.WeightTrainingMHCV(
		chorales['Major'],
		(pitch.Pitch('c4'), pitch.Pitch('g5')),
		5, # k folds
		100, # number of weights to choose
		cd_mh,
		1000, # T
		0.001, # lambda 
		1500, # chain run iter
	)

	output_mh = test_.run()

	pickle.dump(output_mh, open( "output_mh_20180124.p", "wb" ) )


	# gibbs run
	if num == 2:
		cd_gibbs = mh_boolean.create_cross_constraint_dict(
			{
				'NC' : gibbs_boolean.NoCrossing('NC'),
				'SWM' : gibbs_boolean.StepwiseMotion('SWM'),
				'NPI' : gibbs_boolean.NoParallelIntervals('NPI'),
				'OM' : gibbs_boolean.OctaveMax('OM'),
				'NN' : gibbs_boolean.NewNotes('NN'), 
			},
			'Gibbs'
		)

		test_gibbs = ll_method.WeightTrainingGibbsCV(
			chorales['Major'],
			{
				'Alto' : (pitch.Pitch('g3'), pitch.Pitch('c5')),
				'Tenor' : (pitch.Pitch('c3'), pitch.Pitch('e4'))
			 
			},
			5, # k folds
			100, # number of weights to choose
			cd_gibbs,
			1000, # T
			0.001, # lambda 
			1250, # chain run iter
		)

		output_gibbs = test_gibbs.run()

		pickle.dump(output_gibbs, open("output_gibbs_20180124.p", "wb" ) )

if __name__ == '__main__':
   main()
