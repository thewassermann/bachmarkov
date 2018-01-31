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

	# call only if not in memory
	chorales = data_utils.load_clean_chorales()

	cd_mh = mh_boolean.create_cross_constraint_dict({
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

	res = regularization.MHCV(
		np.random.choice(chorales['Major'], size=15),
		5,
		cd_mh,
		[100, 10, 1, .1, .01],
		[100, 10, 1, .1, .01]
	)

	pickle.dump(res, open( "RegularizationCV.p", "wb" ) )

if __name__ == '__main__':
   main()
