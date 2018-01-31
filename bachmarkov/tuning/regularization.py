"""
Stores all Regularization code
"""
import numpy as np
import pandas as pd
import scipy

from tqdm import trange
from tqdm import tqdm

from mh import mh_boolean
from utils import chord_utils, extract_utils, data_utils

from music21 import *

class MCMCMinimizer(mh_boolean.MCMCBooleanSampler):
	"""
	Extension of the MCMCBoolean Sampler to aid regularization tuning
	"""
	
	def split_dicts(self, dict_):
		"""
		Split the constrain dictionary into normal constraints and 
		cross constraints
		"""
		
		cd = {k : dict_[k] for k in list(dict_.keys()) if '/' not in k}
		cross_cd = {k : dict_[k] for k in list(dict_.keys()) if '/' in k}
		return cd, cross_cd
	
	def log_likelihood(self, weights, lambda_1, lambda_2):
		"""
		Calculate the loglikelihood of the chain for each iteration
		"""

		soprano = list(self.soprano.recurse(classFilter=('Note', 'Rest')))
		bass = list(self.bassline.recurse(classFilter=('Note', 'Rest')))

		p_is = np.empty((len(bass), ))

		for j in np.arange(len(bass)):

			# if rest, rest is only choice
			if self.chords[j] == -1 or isinstance(soprano[j], note.Rest) or isinstance(bass[j], note.Rest):
				p_is[j] = 1
				continue

			# possible notes
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(self.chords[j]),
				self.key,
				self.vocal_range
			)

			# loop through notes in chorale
			note_probabilities_dict = {}

			# loop through possible notes
			for k in np.arange(len(possible_notes)):

				possible_chain = soprano[:]
				possible_chain[j] = possible_notes[k]

				# for each note in chorale calculate p_i
				constraints_not_met = \
					[self.constraint_dict[constraint].not_satisfied(self, possible_chain, j) * \
					weights[i] for i, constraint in enumerate(list(self.constraint_dict.keys()))]

				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met))
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}

			p_is[j] = note_probabilities_dict.get(soprano[j].nameWithOctave, 0.001)
			
			
		# match up weights and constraints
		weight_dict = {k : weights[i] for i, k in enumerate(list(self.constraint_dict.keys()))}
		
		wd_, wd_cross = self.split_dicts(weight_dict)

		return -np.nansum(np.log(p_is)) + \
			(lambda_1 * np.nansum(np.array(list(wd_.values()))**2)) + \
			(lambda_2 * np.nansum(np.array(list(wd_cross.values()))**2))


def optimize_chorales(chorales, cd_mh, lambda_1, lambda_2):
	
	res_out = np.empty((len(chorales), len(list(cd_mh.keys()))))
	
	for i in np.arange(len(chorales)):
		
		test_min = MCMCMinimizer(
			chorales[i].parts['Bass'],
			(pitch.Pitch('c4'), pitch.Pitch('g5')),
			chord_utils.degrees_on_beats(chorales[i]),
			cd_mh,
			extract_utils.extract_fermata_layer(
				extract_utils.to_crotchet_stream(chorales[i].parts['Soprano'])
			),
			1., # Starting Temperature
			1., # Cooling Schedule
			0.7, # prob of Local Search
			weight_dict={k : 1 for k in list(cd_mh.keys())},
		)
		
		opt_out = scipy.optimize.minimize(
			test_min.log_likelihood,
			np.ones(len(list(test_min.constraint_dict.keys()))),
			args=(lambda_1, lambda_2),
			method='L-BFGS-B',
			bounds=tuple([(0, None) for x in list(test_min.constraint_dict.keys())]),
		)
		
		res_out[i, :] = opt_out.x
	
	return res_out


def MHCV(chorales, kfolds, cd, lambdas_1, lambdas_2):
	"""
	Function to perform cross-validation on 
	MH constraints
	"""
	# divide chorales into `k` groups
	chorale_index_groups = np.array_split(np.arange(len(chorales)), kfolds)

	CV = np.empty((len(lambdas_1), len(lambdas_2)))

	# loop though folds
	for i, l1 in enumerate(tqdm(lambdas_1, desc='Outer Lambda')):
		for j, l2 in enumerate(tqdm(lambdas_2, desc='Inner Lambda')):
			
			fold_k = np.empty((kfolds, ))
			for k in trange(kfolds, desc='Folds'):
				
				# split into testing and training sets
				testing_ = [chorales[chorale_index] for chorale_index in chorale_index_groups[k]]
				training_indexes = set(np.arange(len(chorales))) - set(chorale_index_groups[k])
				training_ = [chorales[chorale_index] for chorale_index in training_indexes]
				
				train_mses = np.empty((len(training_), len(list(cd.values()))))
				
				for l, train_ in enumerate(training_):
					opt_res = optimize_chorales([train_], cd, l1, l2)
					train_mses[l, :] = opt_res
					
				# average weights
				trained_weights = np.nanmedian(train_mses, axis=0)
				
				# to store absolute difference in likelihoods
				abs_diff_ll = np.empty((len(testing_),))
				
				for m, test_ in enumerate(testing_):
					
					# get optimized weights for true Bach
					bach_optimized_weights = optimize_chorales([test_], cd, l1, l2).flatten()
					
					# get true bach under testing and trained weights
					test_mh = MCMCMinimizer(
						test_.parts['Bass'],
						(pitch.Pitch('c4'), pitch.Pitch('g5')),
						chord_utils.degrees_on_beats(test_),
						cd,
						extract_utils.extract_fermata_layer(
							extract_utils.to_crotchet_stream(test_.parts['Soprano'])
						),
						1., # Starting Temperature
						1., # Cooling Schedule
						0.7, # prob of Local Search
						progress_bar_off=True
					)
					
					test_mh.weight_dict = {k : bach_optimized_weights[i] for i,k in enumerate(list(cd.keys()))}
					bach_opt_weights_ll = test_mh.log_likelihood(bach_optimized_weights, l1, l2)
					test_mh.weight_dict = {k : trained_weights[i] for i,k in enumerate(list(cd.keys()))}
					bach_train_weights_ll = test_mh.log_likelihood(trained_weights, l1, l2)
					
					abs_diff_ll[m] = abs(bach_opt_weights_ll - bach_train_weights_ll)
					
				fold_k[k] = np.nansum(abs_diff_ll)
				
			# cross validation error
			CV[i, j] = np.nansum(fold_k)/len(fold_k)
			
	CV_df = pd.DataFrame(CV)
	CV_df.index = lambdas_1
	CV_df.columns = lambdas_2
	return(CV_df)
