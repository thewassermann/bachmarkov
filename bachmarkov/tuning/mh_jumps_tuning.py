"""
Code to tune parameters relating to MH jump reduction function
"""

import bachmarkov
from utils import chord_utils, extract_utils, data_utils
from hmm import hmm
from mh import mh
import importlib
import networkx as nx
import numpy as np
import pandas as pd
import copy
from music21 import *
from scipy.stats.mstats import gmean

import tqdm

import matplotlib.pyplot as plt


class TuningMH(mh.MH):
	"""
	Extension of MH class used to find the ideal proportions between 
	jump calculation tradeoff. Tuning is done using P1 parameter.

	P1 -> weight for full melody as score function
	"""

	def __init__(self, chorale, vocal_range, fitness_function_dict, P1, prop_chords=None, weight_dict=None):
		self.bassline = extract_utils.to_crotchet_stream(chorale.parts['Bass'])

		if prop_chords is not None:
			self.chords = prop_chords
		else:
			self.chords = chord_utils.degrees_on_beats(chorale) 
		self.vocal_range = vocal_range
		self.key = chorale.analyze('key')
		self.fitness_function_dict = fitness_function_dict
		self.weight_dict = weight_dict
		self.chorale = chorale
		# initialize melody at start of algo
		self.melody = self.init_melody()

		# tuning param
		self.P1 = P1


class NoLargeJumps(mh.FitnessFunction):

	def ff_total_melody(self, melody):
		"""
		See how the proposed note affects the melody as a whole
		"""
		intervals = extract_utils.get_intervals(melody)
		return ((1/len(intervals))*(np.nansum((intervals+0.001)**2)))**-1

	def ff_selected_jump(self, melody, index_):
		"""
		See how the proposed affects the jumps locally
		"""

		if index_ == len(melody) - 1:
			s = -3
			e = -1
		elif index_ == 0:
			s = 0
			e = 2
		else:
			s = index_ - 1
			e = index_ + 1

		intervals = extract_utils.get_intervals(melody, s, e)

		# if interval is greater than a 5th, want to reject
		fifth_comparison = [True if i > 7 else False for i in intervals]
		if any(fifth_comparison):
			return 0.001
		else:
			return ((1/len(intervals))*(np.nansum((intervals+0.001)**2)))**-1

	def ff(self, mh, note_, index_):

		melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))

		# replace original melody note with proposed note
		melody[index_] = copy.deepcopy(note_)

		melody_score = self.ff_total_melody(melody) * mh.P1
		prop_jump_score = self.ff_selected_jump(melody, index_) * (1 - mh.P1)

		return(np.nanmean([melody_score, prop_jump_score]))

	def profiling(self, mh, bass, melody):

		# replace original melody note with proposed note
		intervals = extract_utils.get_intervals(melody)

		return (1/len(intervals))*(np.nansum((intervals+0.001)**2))


def tune_jump_parameters(ps, n_iter, chorale, ax):
	"""
	Function to investigate ideal weighted average between global and local 
	in a particular chorale

	Parameters
	----------

		ps : np.array
			Possible proportions for global weight

		n_iter : int
			number of iterations to run

		chorale : music21.score

		ax : matplotlib.Axes
	"""

	# final squared jumps
	result_array = np.empty(len(ps,))

# loop through proportions
	for i, p in enumerate(ps):
		prop_MH = TuningMH(
			chorale,
			(pitch.Pitch('c4'), pitch.Pitch('g5')),
			{
				'NLJ' : NoLargeJumps('NLJ'),
			},
			P1=p
		)
		result_array[i] = prop_MH.run(n_iter, True, plotting=False).iloc[-1]

	ax.plot(ps, result_array, alpha=0.2, color='blue')
	return result_array

def random_selection(ps, n_iter, chorales, n_chorales, mode):
	"""
	Function to investigate ideal weighted average between global and local 
	in a random number of chorales

	Parameters
	----------

		ps : np.array
			Possible proportions for global weight

		n_iter : int
			number of iterations to run

		n_chorale : list of music21.score

		mode : string
			`Major`/`Minor`
	"""

	# choose chorales to profile
	chorales_ = np.random.choice(chorales[mode], size=n_chorales)

	# set up axes
	#fig, axs = plt.subplots(n_chorales, 1, figsize=(8, n_chorales*5))
	fig, ax = plt.subplots(1, 1, figsize=(10, 8))

	out_array = np.empty((len(ps), n_chorales))

	# loop through chorales
	with tqdm.tqdm(total=n_chorales, desc='Chorales Profiled') as pbar:
		for i, chorale_ in enumerate(chorales_):

			out_array[:, i] = tune_jump_parameters(ps, n_iter, chorale_, ax)
			pbar.update(i)


	# for column in out_array calculate standard deviations in score
	std_ = np.std(out_array, 1)

	# plot summary statistics
	ax.plot(ps, np.mean(out_array, 1), color='red', label='Av. SSI')
	ax.plot(ps, np.mean(out_array, 1) + std_, color='red', ls='--', label='Av. SSI + 1 SD')
	ax.plot(ps, np.mean(out_array, 1) - std_, color='red', ls='--', label='Av. SSI - 1 SD')
	ax.set_xlabel('Proportion of SSIs')
	ax.set_ylabel('SSI')
	ax.set_title('Optimal Porportion for Jumps in {0} chorale after {1} iterations'.format(mode, n_iter))
	ax.legend(loc=2)
	fig.tight_layout()
    