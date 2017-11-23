"""
Class to produce a Hidden Markov Model
"""

import pandas as pd
import numpy as np

import nltk
import nltk.probability as prob

from music21 import *

from utils.chord_utils import degrees_on_beats
from utils.extract_utils import extract_bassline

class HMM():

	def __init__(self, chorales):
		self.beat_chorales, self.transition_matrix =  self.build_transition_matrix_degrees(chorales)
		self.starting_states = self.build_starting_states(self.beat_chorales, self.transition_matrix)
		self.emission_matrix = self.build_emission_probabilities(chorales, self.beat_chorales, self.transition_matrix)
		self.model = self.dfs_to_hmm(self.starting_states, self.emission_matrix, self.transition_matrix)


	def build_transition_matrix_degrees(self, chorales):
		"""
		Function to produce transition matrix between chorales

		Parameters
		----------

			chorales : list of musicxml objects

		Returns
		-------

			transition_matrix : a (n_states x n_states) matrix of
				transition probabilities

		"""
		tmat = pd.DataFrame([])
		rnas = []

		# loop through chorales
		for chorale in chorales:

			# extract roman numeral analysis
			rna = degrees_on_beats(chorale)

			# loop through transitions
			for i in np.arange(len(rna)-1):
			    
		        # if the transition matrix is empty
				if tmat.empty:
					tmat.loc[rna[i], rna[i+1]] = 1
				else:
		            # if chord not seen before add to transition matrix
					if rna[i] not in tmat.index:
						tmat.loc[rna[i], :] = 0
					if rna[i+1] not in tmat.columns:
						tmat.loc[:, rna[i+1]] = 0
		            
		        # add a count for the transition
				tmat.loc[rna[i], rna[i+1]] += 1
		        
		    # return rna for chorales
			rnas.append(rna)
		        
		# now divide each row by the sum of the row to create a stochastic matrix
		rowSums = tmat.sum(axis=1)
		for j in np.arange(len(tmat.index)):
			tmat.iloc[j,:] = tmat.iloc[j,:] / rowSums[j]
		    
		return((rnas, tmat))


	def build_starting_states(self, chorales, tmat):
		"""
		Create dataframe of starting states for HMM

		Parameters
		----------

			chorales : list of musicxml files
				Chorales

			tmat : pd.DataFrame 
				A (n_states x nstates) dataframe of transitions
				between states within the statespace

		Returns
		-------

			starting : pd.Series
				The probability of HMM starting on that state

		"""

		p = pd.Series(0, index=tmat.index)

		# loop through chorales
		for chorale in chorales:
		    
			# get starting states of chorales
			start_state = chorale[0]

			# add count to state
			p[start_state] += 1
		    
		# convert to probability
		p = p/np.sum(p)

		return(p)


	def build_emission_probabilities(self, chorales, beatchorales, tmat):
		"""
		Function to return the emission probabilies for HMM

		Parameters
		----------

			chorales : list of musicxml files
				Chorales

			beatchorales : list of strings
				chords on beats of chorales expressed as relative 
				degrees

			tmat : pd.DataFrame 
				A (n_states x nstates) dataframe of transitions
				between states within the statespace

		Returns
		-------
		"""

		# build empty emission matrix
		em_mat = pd.DataFrame(0, index=tmat.index, columns=np.arange(-1, 12))

		# loop through chorales
		for j, (beatchorale, chorale) in enumerate(zip(beatchorales,chorales)):
		    
			# get bass line and rna for chorale
			bass = extract_bassline(chorale)
			rna = beatchorale
		    
		    # loop through notes/chords
			for i in np.arange(len(bass)):
		        
		        # count emission from a particular chord
				try:
					em_mat.loc[rna[i], bass[i]] += 1
				except IndexError:
					print(j) # errors at 62,96,151
		    
		# make stochastic
		rowSums = em_mat.sum(axis=1)
		for j in np.arange(len(em_mat.index)):
			em_mat.iloc[j,:] = em_mat.iloc[j,:] / rowSums[j]
		    
		return(em_mat)


	def index_dict_to_prob_dict(self, d):
		"""
		Helper function to help format conditional
		distributions for `nltk` package
		"""
		outdict = {}
		for t in d.keys():
			outdict[t] = prob.DictionaryProbDist(d[t])
		return(outdict)


	def dfs_to_hmm(self, starting_df, emission_df, transition_df):
		"""
		Function to convert dataframes into `nltk` HMM format

		Parameters
		----------

			starting_df : pandas.Series
				starting probabilities

			emission_df : pandas.DataFrame
				emission probabilities

			transition_df : pandas.dataframe
				transition probabilities

		Returns
		-------

			HMM: HMM in `nltk` format

		"""

		# to dicts
		em_dict = emission_df.to_dict('index')
		trans_dict = transition_df.to_dict('index')

		# to prob dicts
		em_pdict = self.index_dict_to_prob_dict(em_dict)
		trans_pdict = self.index_dict_to_prob_dict(trans_dict)

		# starting probs
		pstarting=prob.DictionaryProbDist(starting_df.to_dict())

		# to prob distributions
		pem=prob.DictionaryConditionalProbDist(em_pdict)
		pt=prob.DictionaryConditionalProbDist(trans_pdict)

		# hmm
		return nltk.tag.hmm.HiddenMarkovModelTagger(
			symbols=emission_df.columns,
			states=emission_df.index.tolist(),
			transitions=pt,
			outputs=pem,
			priors=pstarting
		)


