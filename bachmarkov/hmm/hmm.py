"""
Class to produce a Hidden Markov Model
"""

import pandas as pd
import numpy as np

from music21 import *

class HMM():

	def __init__(self, chorales):
		beat_chorales, self.transition_matrix =  self.build_transition_matrix_degrees(chorales)
		self.starting_states = self.build_starting_states(beat_chorales, self.transition_matrix)
		self.emission_matrix = self.build_emission_probabilities(chorales, beat_chorales, selftransition_matrix)

	def build_starting_states(chorales, tmat):
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
