"""
Utility functions to do with reading/writing/splitting data
"""

from music21 import *
import os
import pickle
from tqdm import trange
import numpy as np
import sys

"""
These functions are for the reading/writing/cleaning of chorales
"""


def closest_note_not_rest(line, idx, func):
	"""
	Helper function to get the closest non-rest note
	"""

	#if the piece begins/ends on a rest, just use same note
	out = line[idx - 1]

	 # hope to update to nearest note
	for i in np.arange(1, idx):
		if func(idx, i) >= len(line):
			break
		if line[func(idx,i)].isNote and not line[func(idx,i)].isRest:
			out = line[func(idx,i)]
			break

	return out

def partbool(chorales, string):
	"""
	Helper function to ensure that the chorales have the
	correct number and type of parts. Returns False if not
	"""
	prts = chorales[string].parts
	if ((prts[0].partName == 'Soprano') &
		(prts[1].partName == 'Alto') &
		(prts[2].partName == 'Tenor') &
		(prts[3].partName == 'Bass')):
		return True
	else:
		return False


def load_clean_chorales():
	"""
	Function to clean the chorales in the music21 database and save major
	and minor list of musicxml files

	"""
	chor = corpus.getComposer('bach')

	chorales = {}

	# get parsed formats, not just paths [NB: long function (~1min), tqdm used to monitor progress]
	for i in trange(len(chor), desc='# Chorales Parsed'):

	    c = corpus.parse(chor[i])

	    # remove humdrum duplicates
	    if str(c.corpusFilepath)[:-4] != '.krn':
	        chorales[str(c.corpusFilepath)[5:-4]] = c

	fourpart = [c for c in chorales if len(chorales[c].parts) == 4]
	satb = [fp for fp in fourpart if partbool(chorales, fp)]

	satb_major = []
	satb_minor = []

	for i in np.arange(len(satb)):
	    if i == 47: # use this to remove problematic chorales
	        next 
	    elif chorales[satb[i]].analyze('key').mode == 'major':
	        satb_major.append(chorales[satb[i]])
	    elif chorales[satb[i]].analyze('key').mode == 'minor':
	        satb_minor.append(chorales[satb[i]])

	return {
		'Major' : satb_major,
		'Minor' : satb_minor
	}

"""
These functions are for preparing the chorales for modelling processes
"""

def train_test_split(train_pct, chorales):
	"""
	Function to split indices for chorales
	into training and testing sets

	Parameters
	----------

		train_pct : percentage of chorales to be used for
			training set

		chorales : list of musicxml objects

	Returns
	-------

		dictionary of training and testing sets
	"""

	# get training and testing indices
	train_count = int(train_pct * len(chorales))
	train_idx = list(np.random.choice(len(chorales), train_count, replace=False))
	test_idx = list(set(np.arange(len(chorales))).difference(set(train_idx)))

	return {
		'train' : list(np.array(chorales)[train_idx]),
		'test' : list(np.array(chorales)[test_idx])
	}

