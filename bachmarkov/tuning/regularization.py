"""
Stores all Regularization code
"""
import numpy as np
import pandas as pd
import scipy

from tqdm import trange
from tqdm import tqdm

from mh import mh_boolean, mowile
from gibbs import gibbs_boolean
from utils import chord_utils, extract_utils, data_utils

from music21 import *

import multiprocessing as mp

from functools import partial

import cvxpy as cvx
from collections import OrderedDict

class MCMCMinimizer(mh_boolean.MCMCBooleanSampler):
	"""
	Extension of the MCMCBoolean Sampler to aid regularization tuning
	"""

	def split_dicts(self, dict_):
		"""
		Split the constrain dictionary into normal constraints and 
		cross constraints
		"""

		cd = OrderedDict({k : dict_[k] for k in list(dict_.keys()) if '/' not in k})
		cross_cd = OrderedDict({k : dict_[k] for k in list(dict_.keys()) if '/' in k})
		return cd, cross_cd

	def log_likelihood_min(self, weights_single, weights_cross, lambda_1, lambda_2):
		"""
		Calculate the loglikelihood of the chain for each iteration

		"""
		soprano = self.melody
		bass = self.bassline

		p_is = []

		cross_constraint_dict = {}
		for i_, k_ in enumerate(list(self.constraint_dict.keys())):
			for j_, l_ in enumerate(list(self.constraint_dict.keys())):
				if i_ < j_:
					# store `weights_single` indices as tuple 
					# to not double computation
					cross_constraint_dict['{}/{}'.format(k_, l_)] = (i_, j_)

		for j in np.arange(len(bass)):

			# if rest, rest is only choice
			if self.chords[j] == -1 or isinstance(soprano[j], note.Rest) or isinstance(bass[j], note.Rest):
				p_is.append(1)
				continue

			if j == 0:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					None
				)
			else:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					soprano[j-1]
				)

			# be able to convert between prob dist and actual chorale
			note_idx_mapper = np.array([pn.nameWithOctave for pn in possible_notes])
			
			# to store note probabilities
			note_probabilities = []

			# loop through possible notes
			for k in np.arange(len(possible_notes)):

				possible_chain = soprano[:]
				possible_chain[j] = possible_notes[k]

				# list of single constraints
				single_constraints_not_met = np.array([
					self.constraint_dict[constraint].not_satisfied(self, possible_chain, j)
					for constraint in list(self.constraint_dict.keys())
				])
				
				# list of cross constraints
				cross_constraints_not_met = np.array([
					single_constraints_not_met[s_const_1] * single_constraints_not_met[s_const_2]
					for s_const_1, s_const_2 in list(cross_constraint_dict.values())
				])
				
				# these are cvxpy expressions now
				weighted_single_constraints_not_met = \
					sum([single_constraints_not_met[i] * weights_single[i] for i in np.arange(single_constraints_not_met.shape[0])])
				weighted_cross_constraints_not_met = \
					sum([cross_constraints_not_met[i] * weights_cross[i] for i in np.arange(cross_constraints_not_met.shape[0])])
				weighted_constraints = sum([weighted_single_constraints_not_met, weighted_cross_constraints_not_met])

				# non-normalized probability for single notes
				note_probabilities.append(-weighted_constraints)

			# div by totalsum
			if soprano[j].nameWithOctave not in note_idx_mapper:
				p_is.append(0)
			else:
				p_is.append(
					note_probabilities[np.argwhere(note_idx_mapper == soprano[j].nameWithOctave)[0][0]]
				)
		
		return sum([
			sum(p_is),
			lambda_1 * cvx.sum_squares(weights_single),
			lambda_2 * cvx.sum_squares(weights_cross),
		])


	def log_likelihood(self, weights, lambda_1, lambda_2):
		"""
		Calculate the loglikelihood of the chain for each iteration
		"""

		soprano = self.melody
		bass = self.bassline

		p_is = np.empty((len(bass), ))

		for j in np.arange(len(bass)):

			# if rest, rest is only choice
			if self.chords[j] == -1 or isinstance(soprano[j], note.Rest) or isinstance(bass[j], note.Rest):
				p_is[j] = 1
				continue

			if j == 0:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					None
				)
			else:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					soprano[j-1]
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

			p_is[j] = note_probabilities_dict.get(soprano[j].nameWithOctave, 0.01)
			
		# match up weights and constraints
		weight_dict = {k : weights[i] for i, k in enumerate(list(self.constraint_dict.keys()))}
		
		wd_, wd_cross = self.split_dicts(weight_dict)

		return -np.nansum(np.log(p_is)) + \
			(lambda_1 * np.nansum(list(wd_.values()))) + \
			(lambda_2 * np.nansum(list(wd_cross.values())))

	
	
	
def optimize_chorales(chorale, cd_mh, lambda_1, lambda_2, cross_constraints):
	
	test_min = MCMCMinimizer(
		chorale.parts['Bass'],
		(pitch.Pitch('c4'), pitch.Pitch('g5')),
		chord_utils.degrees_on_beats(chorale),
		cd_mh,
		extract_utils.extract_fermata_layer(
			extract_utils.to_crotchet_stream(chorale.parts['Soprano'])
		),
		1., # Starting Temperature
		1., # Cooling Schedule
		0.7, # prob of Local Search,
		1,
		cross_constraints=cross_constraints
	)
		
	weights_single = cvx.Variable(len(test_min.constraint_dict.keys()))

	if cross_constraints:
		cross_constraint_dict = {}
		for i_, k_ in enumerate(list(test_min.constraint_dict.keys())):
			for j_, l_ in enumerate(list(test_min.constraint_dict.keys())):
				if i_ < j_:
					# store `weights_single` indices as tuple 
					# to not double computation
					cross_constraint_dict['{}/{}'.format(k_, l_)] = (i_, j_)
					
		# weights as cvxpy variables (cross constraints)
		weights_cross = cvx.Variable(len(cross_constraint_dict.keys()))

	l1 = cvx.Parameter(1, value=lambda_1, sign="positive")
	l2 = cvx.Parameter(1, value=lambda_2, sign="positive")

	obj = cvx.Minimize(test_min.log_likelihood_min(weights_single, weights_cross, l1, l2))
	const = [weights_single > 0, weights_cross > 0]
	prob = cvx.Problem(obj, const)

	try:
		prob.solve()
		trained_constraints = {k : prob.variables()[0].value[i].item() for i, k in enumerate(list(test_min.constraint_dict.keys()))}
		if cross_constraints:
			trained_constraints.update({k : prob.variables()[1].value[i].item() for i, k in enumerate(list(cross_constraint_dict.keys()))})
		res_out = np.array(list(trained_constraints.values()))
	
	except:
		print('Solver did not Converge for {}'.format(chorale.metadata.title[:-4]))

		if cross_constraints:
			res_out = np.ones(weights_single.size[0] + weights_cross.size[0]) * np.nan
		else:
			res_out = np.ones(weights_single.size[0]) * np.nan

	return res_out


def MHCV(chorales, kfolds, cd, lambdas_1, lambdas_2, cross_constraints):
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
				
				if cross_constraints:
					x_constraints = list(mh_boolean.create_cross_constraint_dict(cd, 'MH', 'No Order'))
					train_mses = np.empty((len(training_), len(x_constraints)))
				else:
					train_mses = np.empty((len(training_), len(list(cd.values()))))
				
				for l, train_ in enumerate(training_):
					opt_res = optimize_chorales(train_, cd, l1, l2, cross_constraints)
					train_mses[l, :] = opt_res
					
				# average weights
				trained_weights = np.nanmedian(train_mses, axis=0)
				
				# to store absolute difference in likelihoods
				abs_diff_ll = np.empty((len(testing_),))
				
				for m, test_ in enumerate(testing_):
					
					# get optimized weights for true Bach
					bach_optimized_weights = optimize_chorales(test_, cd, l1, l2, cross_constraints).flatten()
					
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
						0.7, # prob of Local Search,
						1,
						progress_bar_off=True,
						cross_constraints=cross_constraints
					)
					
					if cross_constraints:
						test_mh.weight_dict = {k : bach_optimized_weights[i] for i,k in enumerate(x_constraints)}
						bach_opt_weights_ll = test_mh.log_likelihood(bach_optimized_weights, l1, l2)

						test_mh.weight_dict = {k : trained_weights[i] for i,k in enumerate(x_constraints)}
						bach_train_weights_ll = test_mh.log_likelihood(trained_weights, l1, l2)
					else:
						test_mh.weight_dict = {k : bach_optimized_weights[i] for i,k in enumerate(list(cd.keys()))}
						bach_opt_weights_ll = test_mh.log_likelihood(bach_optimized_weights, l1, l2)
						test_mh.weight_dict = {k : trained_weights[i] for i,k in enumerate(list(cd.keys()))}
						bach_train_weights_ll = test_mh.log_likelihood(trained_weights, l1, l2)
					
					abs_diff_ll[m] = (bach_opt_weights_ll - bach_train_weights_ll)**2
				fold_k[k] = np.nansum(abs_diff_ll)

			# cross validation error
			CV[i, j] = np.nanmean(fold_k)
			
	CV_df = pd.DataFrame(CV)
	CV_df.index = lambdas_1
	CV_df.columns = lambdas_2
	return(CV_df)

"""
GIBBS
"""

class GibbsMinimizer(gibbs_boolean.GibbsBooleanSampler):
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

	def log_likelihood_min(self, weights_single, weights_cross, lambda_1, lambda_2):
		"""
		Calculate the loglikelihood of the chain for each iteration

		"""
		soprano = self.soprano_flat
		alto = self.alto_flat
		tenor = self.tenor_flat
		bass = self.bass_flat

		p_is = []

		cross_constraint_dict = {}
		for i_, k_ in enumerate(list(self.constraint_dict.keys())):
			for j_, l_ in enumerate(list(self.constraint_dict.keys())):
				if i_ < j_:
					# store `weights_single` indices as tuple 
					# to not double computation
					cross_constraint_dict['{}/{}'.format(k_, l_)] = (i_, j_)

		index_cnt = 0
		part_name = 'Tenor'

		for j in np.arange(2 * len(bass)):

			if part_name == 'Alto':
				part_name = 'Tenor'
			else:
				part_name = 'Alto'

			# if rest, rest is only choice
			if self.chords[index_cnt] == -1 or isinstance(tenor[index_cnt], note.Rest) or isinstance(alto[index_cnt], note.Rest):
				p_is.append(1)
				continue

			if index_cnt == 0:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
					self.key,
					self.vocal_range_dict[part_name],
					None
				)
			else:

				if part_name == 'Alto':
					# possible notes
					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
						self.key,
						self.vocal_range_dict[part_name],
						alto[index_cnt-1]
					)
				else:
					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
						self.key,
						self.vocal_range_dict[part_name],
						tenor[index_cnt-1]
					)

			# be able to convert between prob dist and actual chorale
			note_idx_mapper = np.array([pn.nameWithOctave for pn in possible_notes])

			# to store note probabilities
			note_probabilities = []

			# loop through possible notes
			for k in np.arange(len(possible_notes)):

				if part_name == 'Alto':
					possible_chain = alto[:]
					possible_chain[index_cnt] = possible_notes[k]
				if part_name == 'Tenor':
					possible_chain = tenor[:]
					possible_chain[index_cnt] = possible_notes[k]

				# list of single constraints
				single_constraints_not_met = np.array([
					self.constraint_dict[constraint].not_satisfied(self, possible_chain, index_cnt, part_name)
					for constraint in list(self.constraint_dict.keys())
				])

				# list of cross constraints
				cross_constraints_not_met = np.array([
					single_constraints_not_met[s_const_1] * single_constraints_not_met[s_const_2]
					for s_const_1, s_const_2 in list(cross_constraint_dict.values())
				])

				# these are cvxpy expressions now
				weighted_single_constraints_not_met = \
					sum([single_constraints_not_met[i] * weights_single[i] for i in np.arange(single_constraints_not_met.shape[0])])
				weighted_cross_constraints_not_met = \
					sum([cross_constraints_not_met[i] * weights_cross[i] for i in np.arange(cross_constraints_not_met.shape[0])])
				weighted_constraints = sum([weighted_single_constraints_not_met, weighted_cross_constraints_not_met])

				# non-normalized probability for single notes
				note_probabilities.append(-weighted_constraints)

			# div by totalsum
			if part_name == 'Alto' and alto[index_cnt].nameWithOctave not in note_idx_mapper:
				p_is.append(0)
			elif part_name == 'Tenor' and tenor[index_cnt].nameWithOctave not in note_idx_mapper:
				p_is.append(0)
			else:
				if part_name == 'Alto':
					p_is.append(
						note_probabilities[np.argwhere(note_idx_mapper == alto[index_cnt].nameWithOctave)[0][0]]
					)
				else:
					p_is.append(
						note_probabilities[np.argwhere(note_idx_mapper == tenor[index_cnt].nameWithOctave)[0][0]]
					)    

			if j % 2 == 1:
				index_cnt += 1 

		return sum([
			sum(p_is),
			lambda_1 * cvx.sum_squares(weights_single),
			lambda_2 * cvx.sum_squares(weights_cross),
		])


	def log_likelihood(self, weights, lambda_1, lambda_2):
		"""
		Compute loglikelihood of inner parts
		"""
		soprano = self.soprano_flat
		alto = self.alto_flat
		tenor = self.tenor_flat
		bass = self.bass_flat

		p_is = np.empty((len(tenor * 2),))

		index_cnt = 0
		part_name = 'Tenor'

		for j in np.arange(len(tenor * 2)):

			# index flag and previous part determine which part to operate on next
			if part_name == 'Alto':
				part_name = 'Tenor'
			else:
				part_name = 'Alto'

			# if rest, rest is only choice
			if self.chords[index_cnt] == -1 or isinstance(tenor[index_cnt], note.Rest) or isinstance(alto[index_cnt], note.Rest):
				p_is[j] = 1
				continue

			if index_cnt == 0:
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
					self.key,
					self.vocal_range_dict[part_name],
					None
				)
			else:

				if part_name == 'Alto':
					# possible notes
					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
						self.key,
						self.vocal_range_dict[part_name],
						alto[index_cnt-1]
					)
				else:
					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
						self.key,
						self.vocal_range_dict[part_name],
						tenor[index_cnt-1]
					)

			# loop through notes in chorale
			note_probabilities_dict = {}

			# loop through possible notes alto then tenor 
			for k in np.arange(len(possible_notes)):

				if part_name == 'Alto':
					possible_chain = alto[:]
					possible_chain[index_cnt] = possible_notes[k]
				if part_name == 'Tenor':
					possible_chain = tenor[:]
					possible_chain[index_cnt] = possible_notes[k]

				# for each note in chorale calculate p_i
				constraints_not_met = \
					[self.constraint_dict[constraint].not_satisfied(self, possible_chain, index_cnt, part_name) * \
					weights[i] for i, constraint in enumerate(list(self.constraint_dict.keys()))]

				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met))
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}


			if part_name == 'Alto':
				p_is[j] = note_probabilities_dict.get(alto[index_cnt].nameWithOctave, 0.01)
			else:
				p_is[j] = note_probabilities_dict.get(tenor[index_cnt].nameWithOctave, 0.01)

			if j % 2 == 1:
				index_cnt += 1 
				
		# match up weights and constraints
		weight_dict = {k : weights[i] for i, k in enumerate(list(self.constraint_dict.keys()))}
		
		wd_, wd_cross = self.split_dicts(weight_dict)

		# return scores
		return -np.nansum(np.log(p_is)) + \
			(lambda_1 * np.nansum(list(wd_.values()))) + \
			(lambda_2 * np.nansum(list(wd_cross.values())))

def optimize_chorales_gibbs(chorale, cd_gibbs, lambda_1, lambda_2):

	test_min = GibbsMinimizer(
		chorale.parts['Bass'],
		{
			'Alto' : (pitch.Pitch('g3'), pitch.Pitch('c5')),
			'Tenor' : (pitch.Pitch('c3'), pitch.Pitch('e4')), 
		},
		chord_utils.degrees_on_beats(chorale),
		chorale.parts['Soprano'],
		cd_gibbs,
		extract_utils.extract_fermata_layer(
			extract_utils.to_crotchet_stream(chorale.parts['Soprano'])
		),
		1,
		1,
		.9,
		1,
		progress_bar_off=True,
	)

	weights_single = cvx.Variable(len(test_min.constraint_dict.keys()))

	cross_constraint_dict = {}
	for i_, k_ in enumerate(list(test_min.constraint_dict.keys())):
		for j_, l_ in enumerate(list(test_min.constraint_dict.keys())):
			if i_ < j_:
				# store `weights_single` indices as tuple 
				# to not double computation
				cross_constraint_dict['{}/{}'.format(k_, l_)] = (i_, j_)

	# weights as cvxpy variables (cross constraints)
	weights_cross = cvx.Variable(len(cross_constraint_dict.keys()))

	l1 = cvx.Parameter(1, value=lambda_1, sign="positive")
	l2 = cvx.Parameter(1, value=lambda_2, sign="positive")

	obj = cvx.Minimize(test_min.log_likelihood_min(weights_single, weights_cross, l1, l2))
	const = [weights_single > 0, weights_cross > 0]
	prob = cvx.Problem(obj, const)

	try:
		prob.solve()
		trained_constraints = {k : prob.variables()[0].value[i].item() for i, k in enumerate(list(test_min.constraint_dict.keys()))}
		trained_constraints.update({k : prob.variables()[1].value[i].item() for i, k in enumerate(list(cross_constraint_dict.keys()))})
		res_out = np.array(list(trained_constraints.values()))

	except:
		print('Solver did not Converge for {}'.format(chorale.metadata.title[:-4]))

		res_out = np.ones(weights_single.size[0] + weights_cross.size[0]) * np.nan

	return res_out


def MHCVGibbs(chorales, kfolds, cd, lambdas_1, lambdas_2):
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

				x_constraints = list(mh_boolean.create_cross_constraint_dict(cd, 'Gibbs', 'No Order'))
				train_mses = np.empty((len(training_), len(x_constraints)))

				for l, train_ in enumerate(training_):
					opt_res = optimize_chorales_gibbs(train_, cd, l1, l2)
					train_mses[l, :] = opt_res

				# average weights
				trained_weights = np.nanmedian(train_mses, axis=0)

				# to store absolute difference in likelihoods
				sq_diff_ll = np.empty((len(testing_),))

				for m, test_ in enumerate(testing_):

					# get optimized weights for true Bach
					bach_optimized_weights = optimize_chorales_gibbs(test_, cd, l1, l2).flatten()

					# get true bach under testing and trained weights
					test_gibbs = GibbsMinimizer(
						test_.parts['Bass'],
						{
							'Alto' : (pitch.Pitch('g3'), pitch.Pitch('c5')),
							'Tenor' : (pitch.Pitch('c3'), pitch.Pitch('e4')), 
						},
						chord_utils.degrees_on_beats(test_),
						test_.parts['Soprano'],
						cd,
						extract_utils.extract_fermata_layer(
							extract_utils.to_crotchet_stream(test_.parts['Soprano'])
						),
						1,
						1,
						.9,
						1,
						progress_bar_off=True,
					)

					test_gibbs.weight_dict = {k : bach_optimized_weights[i] for i,k in enumerate(x_constraints)}
					bach_opt_weights_ll = test_gibbs.log_likelihood(bach_optimized_weights, l1, l2)

					test_gibbs.weight_dict = {k : trained_weights[i] for i,k in enumerate(x_constraints)}
					bach_train_weights_ll = test_gibbs.log_likelihood(trained_weights, l1, l2)

					sq_diff_ll[m] = (bach_opt_weights_ll - bach_train_weights_ll)**2
					
				fold_k[k] = np.nansum(sq_diff_ll)

			# cross validation error
			CV[i, j] = np.nanmean(fold_k)

	CV_df = pd.DataFrame(CV)
	CV_df.index = lambdas_1
	CV_df.columns = lambdas_2
	return(CV_df)


