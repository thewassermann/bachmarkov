import bachmarkov
from utils import chord_utils, extract_utils, data_utils
from hmm import hmm
from mh import mh, mh_boolean
from gibbs import gibbs_boolean
from tuning import convergence_diagnostics
import importlib
import networkx as nx
import numpy as np
import pandas as pd
import copy
from music21 import *
from scipy.stats.mstats import gmean

import seaborn as sns

import tqdm
from tqdm import trange

class WeightTrainer():
	"""
	Given a list of constraints, calculate corresponding weights that minimize the log-likelihood function in the
	true Bach soprano line
	"""
	
	def __init__(self, vocal_range, soprano, bass, chords, constraint_dict, T_0, lambda_):
		
		self.vocal_range = vocal_range
		self.soprano = list(soprano.recurse(classFilter=('Note','Rest')))
		self.bass = list(bass.recurse(classFilter=('Note', 'Rest')))
		self.bassline = bass
		self.chords = chords
		self.constraint_dict = constraint_dict
		self.T = T_0
		self.lambda_ = lambda_
		
		self.key = bass.analyze('key')
		
	def run(self, n_iter):
		
		scores = np.empty((n_iter, len(self.constraint_dict.keys()) + 1))
		scores_df = pd.DataFrame(scores)
		scores_df.index = np.arange(n_iter)
		scores_df.columns = list(self.constraint_dict.keys()) + ['Log-Lik']
		
		for i in trange(n_iter, desc=' Number of Weightings tried', disable=True):
		
			# generate random weight metric
			random_ws = np.random.exponential(100, size=len(list(self.constraint_dict.keys())))
			weight_dict = {k: random_ws[iter_] for iter_, k in enumerate(list(self.constraint_dict.keys()))}
			scores_df.iloc[i, :-1] = np.array(list(weight_dict.values()))
			
			p_is = np.empty((len(self.bass),))
			
			for j in np.arange(len(self.bass)):
				
				# if rest, rest is only choice
				if self.chords[j] == -1:
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
					
					possible_chain = self.soprano[:]
					possible_chain[j] = possible_notes[k]
		
					# for each note in chorale calculate p_i
					constraints_not_met = \
						[self.constraint_dict[constraint].not_satisfied(self, possible_chain, j) * \
						weight_dict[constraint] for constraint in list(self.constraint_dict.keys())]
					
					note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met)/self.T)
					total_sum = np.nansum(list(note_probabilities_dict.values()))
					note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}
				

				p_is[j] = note_probabilities_dict.get(self.soprano[j].nameWithOctave, 0.001)
		
			# return scores
			scores_df.loc[i, 'Log-Lik'] = -np.log((np.nanprod(p_is))) + (self.lambda_ * np.nansum(random_ws**2))
		
		return scores_df
		

class WeightTrainerInner():
	"""
	Given a list of constraints, calculate corresponding weights that minimize the log-likelihood function in the
	true Bach inner parts line
	"""
	
	def __init__(self, vocal_range_dict, soprano, alto, tenor, bass, chords, constraint_dict, T_0, lambda_):
		
		self.vocal_range_dict = vocal_range_dict
		self.soprano = soprano
		self.alto = alto
		self.tenor = tenor
		self.bass = bass
		self.bassline = bass
		self.chords = chords
		self.constraint_dict = constraint_dict
		self.T = T_0
		self.lambda_ = lambda_
		
		self.key = bass.analyze('key')
		
	def run(self, n_iter):
		
		scores = np.empty((n_iter, len(self.constraint_dict.keys()) + 1))
		scores_df = pd.DataFrame(scores)
		scores_df.index = np.arange(n_iter)
		scores_df.columns = list(self.constraint_dict.keys()) + ['Log-Lik']
		
		
		flat_alto = list(self.alto.recurse(classFilter=('Note', 'Rest')))
		flat_tenor = list(self.tenor.recurse(classFilter=('Note', 'Rest')))
		
		for i in trange(n_iter, desc=' Number of Weightings tried', disable=True):
			
			# generate random weight metric
			random_ws = np.random.exponential(100, size=len(list(self.constraint_dict.keys())))
			weight_dict = {k: random_ws[iter_] for iter_, k in enumerate(list(self.constraint_dict.keys()))}
			scores_df.iloc[i, :-1] = np.array(list(weight_dict.values()))
			
			p_is = np.empty((len(flat_tenor * 2),))
			
			index_cnt = 0
			part_name = 'Tenor'
			
			for j in np.arange(len(flat_tenor * 2)):
				
				# index flag and previous part determine which part to operate on next
				if part_name == 'Alto':
					part_name = 'Tenor'
				else:
					part_name = 'Alto'
				
				# if rest, rest is only choice
				if self.chords[index_cnt] == -1:
					p_is[j] = 1
					continue
				
				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
					self.key,
					self.vocal_range_dict[part_name]
				)
				
				# loop through notes in chorale
				note_probabilities_dict = {}
				
				# loop through possible notes alto then tenor 
				for k in np.arange(len(possible_notes)):
					
					if part_name == 'Alto':
						possible_chain = flat_alto[:]
						possible_chain[index_cnt] = possible_notes[k]
					if part_name == 'Tenor':
						possible_chain = flat_tenor[:]
						possible_chain[index_cnt] = possible_notes[k]
		
					# for each note in chorale calculate p_i
					constraints_not_met = \
						[self.constraint_dict[constraint].not_satisfied(self, possible_chain, index_cnt, part_name) * \
						weight_dict[constraint] for constraint in list(self.constraint_dict.keys())]
					
					note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met)/self.T)
					total_sum = np.nansum(list(note_probabilities_dict.values()))
					note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}
				
					
				if part_name == 'Alto':
					p_is[j] = note_probabilities_dict.get(flat_alto[index_cnt].nameWithOctave, 0.01)
				else:
					p_is[j] = note_probabilities_dict.get(flat_tenor[index_cnt].nameWithOctave, 0.01)
				
				if j % 2 == 1:
					index_cnt += 1 
		
			# return scores
			scores_df.loc[i, 'Log-Lik'] = -np.nansum(np.log(p_is)) #+ (self.lambda_ * np.nansum(list(weight_dict.values())**2))
		
		return scores_df


def weight_aggregator(chorales, constraint_dict, vocal_range, n_iter):
	"""
	Function to train weights over a list of chorales. Each chrorale 
	undergoes a WeightTrainer run for `n_iter` iterations and returns the minimum log likelihood
	"""
	
	scores = np.empty((len(chorales), len(constraint_dict.keys()) + 1))
	scores_df = pd.DataFrame(scores)
	scores_df.index = np.arange(len(chorales))
	scores_df.columns = list(constraint_dict.keys()) + ['Log-Lik']
	
	for i, chorale in enumerate(chorales):
		
		chorale_test = WeightTrainer(
			vocal_range,
			extract_utils.to_crotchet_stream(chorale.parts['Soprano']),
			extract_utils.to_crotchet_stream(chorale.parts['Bass']),
			chord_utils.degrees_on_beats(chorale),
			constraint_dict,
			1000,
			0.96
		)

		try:
			chorale_test_out = chorale_test.run(n_iter)
		
			min_log_lik = np.array(chorale_test_out.iloc[chorale_test_out['Log-Lik'].argmax()])
			scores_df.iloc[i, :] = min_log_lik
		except:
			scores_df.iloc[i, :] = np.nan
		
		
	return scores_df


def weight_aggregator_inner(chorales, constraint_dict, vocal_range_dict, n_iter):
	"""
	Function to train weights over a list of chorales. Each chrorale 
	undergoes a WeightTrainer run for `n_iter` iterations and returns the minimum log likelihood
	"""
	
	scores = np.empty((len(chorales), len(constraint_dict.keys()) + 1))
	scores_df = pd.DataFrame(scores)
	scores_df.index = np.arange(len(chorales))
	scores_df.columns = list(constraint_dict.keys()) + ['Log-Lik']
	
	for i, chorale in enumerate(chorales):
		
		chorale_test = WeightTrainerInner(
			vocal_range_dict,
			extract_utils.to_crotchet_stream(chorale.parts['Soprano']),
			extract_utils.to_crotchet_stream(chorale.parts['Alto']),
			extract_utils.to_crotchet_stream(chorale.parts['Tenor']),
			extract_utils.to_crotchet_stream(chorale.parts['Bass']),
			chord_utils.degrees_on_beats(chorale),
			constraint_dict,
			1000,
			0.96
		)

		try:
			chorale_test_out = chorale_test.run(n_iter)
		
			min_log_lik = np.array(chorale_test_out.iloc[chorale_test_out['Log-Lik'].argmax()])
			scores_df.iloc[i, :] = min_log_lik
		except:
			scores_df.iloc[i, :] = np.nan
		
		
	return scores_df


class WeightTrainingMHCV():
	"""
	Class to take a list of chorales and perform a k-Fold
	cross validation in order to train the weights
	
	Parameters
	----------
	
		chorales : list of music.Score
		vocal_range : tuple of music21.pitches
		kfolds : int
		n_iter : int
			number of weights to try in order to select best-performing
		cd : dict of mh_boolean.Constraints
		T : float
		lambda_ : float
			regularization parameter
	
	"""
	
	def __init__(self, chorales, vocal_range, kfolds, n_iter, cd, T, lambda_, N):
		self.chorales = chorales
		self.vocal_range = vocal_range
		self.kfolds = kfolds
		self.n_iter = n_iter
		self.cd = cd
		self.T = T
		self.lambda_ = lambda_
		self.N = N
	
	def run(self):
		"""
		Run the cross validation
		"""
		
		# divide chorales into `k` groups
		chorale_index_groups = np.array_split(np.arange(len(self.chorales)), self.kfolds)

		fold_weights_and_MSE = np.empty((self.kfolds, len(list(self.cd.values())) + 1))
		
		# loop through cv folds
		for i in trange(self.kfolds, desc='iter-fold validation'):

			# split into testing and training sets
			testing_ = [self.chorales[chorale_index] for chorale_index in chorale_index_groups[i]]
			training_indexes = set(np.arange(len(self.chorales))) - set(chorale_index_groups[i])
			training_ = [self.chorales[chorale_index] for chorale_index in training_indexes]

			# structure to store best weights
			best_weights_train = np.empty((len(training_), len(list(self.cd.values())) + 1))

			# iterate through chorales
			for j in np.arange(len(training_)):

				mh_trainer = WeightTrainer(
					self.vocal_range,
					extract_utils.to_crotchet_stream(training_[j].parts['Soprano']),
					extract_utils.to_crotchet_stream(training_[j].parts['Bass']),
					chord_utils.degrees_on_beats(training_[j]),
					self.cd,
					self.T,
					self.lambda_,
				)
				weights_out = mh_trainer.run(self.n_iter)
				best_weights_train[j, :] = weights_out.iloc[np.argmin(weights_out['Log-Lik']), :]

			# calculate most appropriate weight dictionary
			best_weights_train = pd.Series(np.nanmean(best_weights_train, axis=0)[:-1], index=list(self.cd.keys()))
			
			print(best_weights_train)
			
			test_logliks = np.empty((len(testing_), 2))
			for k in np.arange(len(testing_)):
			
				# desired weight dict
				wd = {k_ : best_weights_train.loc[k_] for k_ in best_weights_train.index}
			
				# run a mh_boolean chain for `N` iteration
				testing_mh_out = mh_boolean.MCMCBooleanSampler(
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					self.vocal_range,
					chord_utils.degrees_on_beats(testing_[k]),
					self.cd,
					extract_utils.extract_fermata_layer(
						extract_utils.to_crotchet_stream(testing_[k].parts['Soprano'])
					),
					self.T, # Starting Temperature
					.923, # Cooling Schedule
					0., # prob of Local Search
					weight_dict = wd
				)
				
				# run for `N` iterations
				testing_mh_out.run(self.N, False, False)
			
				# calculate loglikelihood on outputs
				test_logliks[k, 0] = self.log_likelihood(
					testing_mh_out,
					testing_mh_out.soprano,
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					chord_utils.degrees_on_beats(testing_[k]),
					testing_[k].analyze('key'),
					wd
				)
				
				# true loglik
				test_logliks[k, 1] = self.log_likelihood(
					testing_mh_out,
					extract_utils.to_crotchet_stream(testing_[k].parts['Soprano']),
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					chord_utils.degrees_on_beats(testing_[k]),
					testing_[k].analyze('key'),
					wd
				)
				
			print(test_logliks)

			av_mse = (np.nansum((test_logliks[:, 0] - test_logliks[:, 1])**2)) / len(testing_)
			
			fold_weights_and_MSE[i, :-1] = np.array(list(wd.values()))
			fold_weights_and_MSE[i, -1] = av_mse
			
		return pd.DataFrame(fold_weights_and_MSE, columns=list(self.cd.keys()) + ['MSE'])

	def log_likelihood(self, model, soprano, bass, chords, key, weight_dict):
		"""
		Calculate the loglikelihood of the chain for each iteration
		"""

		soprano = list(soprano.recurse(classFilter=('Note', 'Rest')))
		bass = list(bass.recurse(classFilter=('Note', 'Rest')))

		p_is = np.empty((len(bass), ))

		for j in np.arange(len(bass)):

			# if rest, rest is only choice
			if chords[j] == -1 or isinstance(soprano[j], note.Rest) or isinstance(bass[j], note.Rest):
				p_is[j] = 1
				continue

			# possible notes
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(chords[j]),
				key,
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
					[self.cd[constraint].not_satisfied(model, possible_chain, j) * \
					weight_dict[constraint] for constraint in list(self.cd.keys())]

				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met))
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}

			p_is[j] = note_probabilities_dict.get(soprano[j].nameWithOctave, 0.001)

		return -np.nansum(np.log(p_is))


class WeightTrainingGibbsCV():
	"""
	Class to take a list of chorales and perform a k-Fold
	cross validation in order to train the weights
	
	Parameters
	----------
	
		chorales : list of music.Score
		vocal_range : tuple of music21.pitches
		kfolds : int
		n_iter : int
			number of weights to try in order to select best-performing
		cd : dict of gibbs_boolean.Constraints
		T : float
		lambda_ : float
			regularization parameter
	
	"""
	
	def __init__(self, chorales, vocal_range_dict, kfolds, n_iter, cd, T, lambda_, N):
		self.chorales = chorales
		self.vocal_range_dict = vocal_range_dict
		self.kfolds = kfolds
		self.n_iter = n_iter
		self.cd = cd
		self.T = T
		self.lambda_ = lambda_
		self.N = N
	
	def run(self):
		"""
		Run the cross validation
		"""
		
		# divide chorales into `k` groups
		chorale_index_groups = np.array_split(np.arange(len(self.chorales)), self.kfolds)

		fold_weights_and_MSE = np.empty((self.kfolds, len(list(self.cd.values())) + 1))
		
		# loop through cv folds
		for i in trange(self.kfolds, desc='iter-fold validation'):

			# split into testing and training sets
			testing_ = [self.chorales[chorale_index] for chorale_index in chorale_index_groups[i]]
			training_indexes = set(np.arange(len(self.chorales))) - set(chorale_index_groups[i])
			training_ = [self.chorales[chorale_index] for chorale_index in training_indexes]

			# structure to store best weights
			best_weights_train = np.empty((len(training_), len(list(self.cd.values())) + 1))

			# iterate through chorales
			for j in np.arange(len(training_)):

				gibbs_trainer = WeightTrainerInner(
					self.vocal_range_dict,
					extract_utils.to_crotchet_stream(training_[j].parts['Soprano']),
					extract_utils.to_crotchet_stream(training_[j].parts['Alto']),
					extract_utils.to_crotchet_stream(training_[j].parts['Tenor']),
					extract_utils.to_crotchet_stream(training_[j].parts['Bass']),
					chord_utils.degrees_on_beats(training_[j]),
					self.cd,
					self.T,
					self.lambda_,
				)
				weights_out = gibbs_trainer.run(self.n_iter)
				best_weights_train[j, :] = weights_out.iloc[np.argmin(weights_out['Log-Lik']), :]

			# calculate most appropriate weight dictionary
			best_weights_train = pd.Series(np.nanmean(best_weights_train, axis=0)[:-1], index=list(self.cd.keys()))
			
			print(best_weights_train)
			
			test_logliks = np.empty((len(testing_), 2))
			for k in np.arange(len(testing_)):
			
				# desired weight dict
				wd = {k_ : best_weights_train.loc[k_] for k_ in best_weights_train.index}
			
				# run a gibbs_boolean chain for `N` iteration
				testing_gibbs_out = gibbs_boolean.GibbsBooleanSampler(
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					self.vocal_range_dict,
					chord_utils.degrees_on_beats(testing_[k]),
					extract_utils.to_crotchet_stream(testing_[k].parts['Soprano']),
					self.cd,
					extract_utils.extract_fermata_layer(
						extract_utils.to_crotchet_stream(testing_[k].parts['Soprano'])
					),
					self.T, # Starting Temperature
					.923, # Cooling Schedule
					0., # prob of Local Search
					weight_dict = wd
				)
				
				# run for `N` iterations
				testing_gibbs_out.run(self.N, False, False)
			
				# calculate loglikelihood on outputs
				test_logliks[k, 0] = self.log_likelihood(
					testing_gibbs_out,
					testing_gibbs_out.soprano,
					testing_gibbs_out.alto,
					testing_gibbs_out.tenor,
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					chord_utils.degrees_on_beats(testing_[k]),
					testing_[k].analyze('key'),
					wd
				)
				
				# true loglik
				test_logliks[k, 1] = self.log_likelihood(
					testing_gibbs_out,
					extract_utils.to_crotchet_stream(testing_[k].parts['Soprano']),
					extract_utils.to_crotchet_stream(testing_[k].parts['Alto']),
					extract_utils.to_crotchet_stream(testing_[k].parts['Tenor']),
					extract_utils.to_crotchet_stream(testing_[k].parts['Bass']),
					chord_utils.degrees_on_beats(testing_[k]),
					testing_[k].analyze('key'),
					wd
				)
				
			print(test_logliks)

			av_mse = (np.nansum((test_logliks[:, 0] - test_logliks[:, 1])**2)) / len(testing_)
			
			fold_weights_and_MSE[i, :-1] = np.array(list(wd.values()))
			fold_weights_and_MSE[i, -1] = av_mse
			
		return pd.DataFrame(fold_weights_and_MSE, columns=list(self.cd.keys()) + ['MSE'])

	
	def log_likelihood(self, model, soprano, alto, tenor, bass, chords, key, weight_dict):
		"""
		Compute loglikelihood of inner parts
		"""
		
		soprano = list(soprano.recurse(classFilter=('Note', 'Rest')))
		flat_alto = list(alto.recurse(classFilter=('Note', 'Rest')))
		flat_tenor = list(tenor.recurse(classFilter=('Note', 'Rest')))
		bass = list(bass.recurse(classFilter=('Note', 'Rest')))

		p_is = np.empty((len(flat_tenor * 2),))

		index_cnt = 0
		part_name = 'Tenor'

		for j in np.arange(len(flat_tenor * 2)):

			# index flag and previous part determine which part to operate on next
			if part_name == 'Alto':
				part_name = 'Tenor'
			else:
				part_name = 'Alto'

			# if rest, rest is only choice
			if chords[index_cnt] == -1 or isinstance(soprano[index_cnt], note.Rest) or isinstance(bass[index_cnt], note.Rest):
				p_is[j] = 1
				continue

			# possible notes
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(chords[index_cnt]),
				key,
				model.vocal_range_dict[part_name]
			)

			# loop through notes in chorale
			note_probabilities_dict = {}

			# loop through possible notes alto then tenor 
			for k in np.arange(len(possible_notes)):

				if part_name == 'Alto':
					possible_chain = flat_alto[:]
					possible_chain[index_cnt] = possible_notes[k]
				if part_name == 'Tenor':
					possible_chain = flat_tenor[:]
					possible_chain[index_cnt] = possible_notes[k]

				# for each note in chorale calculate p_i
				constraints_not_met = \
					[self.cd[constraint].not_satisfied(model, possible_chain, index_cnt, part_name) * \
					weight_dict[constraint] for constraint in list(self.cd.keys())]

				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met)/self.T)
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}


			if part_name == 'Alto':
				p_is[j] = note_probabilities_dict.get(flat_alto[index_cnt].nameWithOctave, 0.01)
			else:
				p_is[j] = note_probabilities_dict.get(flat_tenor[index_cnt].nameWithOctave, 0.01)

			if j % 2 == 1:
				index_cnt += 1 

		# return scores
		return -np.nansum(np.log(p_is))
		

