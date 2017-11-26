"""
Function for evaluating the efficacy of models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
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
	try :
		df = pd.DataFrame.from_dict({'Pred': pred, 'Obs' : obs, 'Bass' : bass})
		return df
	except:
		print(chorale)
		return None


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
	if compare_df is None:
		return np.nan
	else:
		try:
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
			return np.mean(out_)
		except:
			return np.nan


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

	accuracies = []

	for i, chorale in enumerate(chorales):
			cmp_df = compare_chords(hmm, chorale)
			if cmp_df is not None:
				accuracies.append(accuracy_chords(cmp_df))

	return np.array(accuracies)


def plot_chord_study(cas_output, title):
		"""
		Function to plot results of chord accuracy study

		Parameters
		----------

			cas_output : numpy.array
				outpit of chord accuracy study

		Returns
		-------

			plot results 
		"""

		fig, (ax_hist, ax_stats) = plt.subplots(1,2, figsize=(10, 4))

		# histogram axis
		sns.distplot(cas_output[~np.isnan(cas_output)], ax=ax_hist)

		# summary index
		summary_index = [
			'N',
			'Min',
			'Median',
			'Mean',
			'Max'
		]

		summary_data = [
			len(cas_output),
			np.nanmin(cas_output),
			np.nanmedian(cas_output),
			np.nanmean(cas_output),
			np.nanmax(cas_output),
		]

		summary_data = ['%1.3f' % x for x in summary_data]
		summary_df = pd.DataFrame(summary_data, index=summary_index)

		ax_stats.axis('off')
		ax_stats.axis('tight')
		tbl = ax_stats.table(
			cellText=summary_df.values,
			rowLabels=summary_df.index,
			loc='center'
		)
		tbl.scale(1, 1.5)

		fig.suptitle(title)
	

