"""
Function for evaluating the efficacy of models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.chord_utils import degrees_on_beats
from utils.extract_utils import extract_bassline


def compare_chords(hmm, chorale):
	"""
	Compare the chord model predicted and observed

	Parameters
	----------

		hmm : nltk tagger object
			model to test

		chorale : musicxml

	Returns
	-------

		Dataframe with `Pred`, `Obs` and `Bass` columns
	"""
	bass = extract_bassline(chorale)
	pred = hmm.best_path(bass)
	obs = degrees_on_beats(chorale)
	return pd.DataFrame.from_dict({'Pred': pred, 'Obs' : obs, 'Bass' : bass})


def accuracy_chords(compare_df):
	"""
	Function to quantify accuracy of chord prediction model

	Parameters
	----------

		compare_df : pandas.DataFrame
			Output of `compare_chords`

	Returns
	-------

		Float representing accuracy
	"""

	# array to store output
	out_ = np.empty(len(compare_df.index),)

	# loop through predictions
	for i in np.arange(len(compare_df.index)):

    	# if rests
		if compare_df.loc[i, 'Obs'] == -1:
			if compare_df.loc[i, 'Pred'] == -1:
				out_[i] = 1.
			else:
				out_[i] = 0.
		else:
			# divide chord into degrees handle exception if only one note
			if len(compare_df.loc[i, 'Obs']) == 1:
				obs = set(compare_df.loc[i, 'Obs'])
			else:
				obs = set(compare_df.loc[i, 'Obs'].split('/'))
			if len(compare_df.loc[i, 'Pred']) == 1:
				pred = set(compare_df.loc[i, 'Pred'])
			else:
				pred = set(compare_df.loc[i, 'Pred'].split('/'))
			num = len(obs)
			corr = 0.

            # predicted
			for p in compare_df.loc[i, 'Pred']:
				if p in obs:
					corr += 1.

			out_[i] = corr / num
            
	return(np.mean(out_))


def chord_accuracy_study(hmm, chorales):
	"""
	Function to take in list of chords and present diagnostics
	on the model's efficacy over that cohort

	Parameters
	----------

		hmm : `nltk` tagger

		chorales : list of musicxml objects

	Returns
	-------

		Summary data on accuracy

	"""

	accuracies = np.empty(len(chorales),)

	for i, chorale in enumerate(chorales):
		try:
			cmp_df = compare_chords(hmm, chorale)
		except:
			print(chorale)
		try:
			accuracies[i] = accuracy_chords(cmp_df)
		except:
			print(chorale)

	pd.Series(accuracies).hist()
	

