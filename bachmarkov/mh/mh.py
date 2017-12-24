"""
File to store functions/classes for 
Metropolis-Hastings based algorithms
"""

from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2

from tqdm import trange
import matplotlib.pyplot as plt

import copy

from utils import chord_utils, extract_utils, data_utils

class MH():
	"""
	A class to perform tasks relating
	to the Metropolis Hastings Algorithm.

	Different implementations of this algorithm should inherit 
	from this class
	"""

	def __init__(self, chorale, vocal_range, fitness_function_dict, weight_dict=None):
		self.bassline = extract_utils.to_crotchet_stream(chorale.parts['Bass'])
		self.chords = chord_utils.degrees_on_beats(chorale) 
		self.vocal_range = vocal_range
		self.key = chorale.analyze('key')
		self.fitness_function_dict = fitness_function_dict
		self.weight_dict = weight_dict

		# initialize melody at start of algo
		self.melody = self.init_melody()


	def random_note_in_range(self):
		"""
		Function to return a random note in the desired vocal range
		"""

		# get vocal range
		low, high = self.vocal_range
		low_high_int = interval.Interval(noteStart=low, noteEnd=high)
		num_semitones =  low_high_int.semitones

		low_note = note.Note(low.nameWithOctave, quarterLength=1)

		# pick random semitone in range
		semitone_select = np.random.randint(1, num_semitones)
		return low_note.transpose(semitone_select)



	def init_melody(self):
		"""
		Function to initialize melody line with 
		random sequence of notes within a vocal range

		Parameters
		---------

		    bass_on_beat : music21.part
		        bassline of a chorale parsed so
		        only the notes on beats are included

		    vocal_range : tuple of music21.pitch
		        The highest and lowest note that can
		        be sung by the vocal part to be filled in

		Returns
		-------

		    out_stream : music21.stream
		        part initialized to random note in the range

		"""
		out_stream = stream.Stream()

		# loop through measures
		for el in self.bassline.recurse(skipSelf=True):
			if el == clef.BassClef():
				out_stream.insert(clef.TrebleClef())
			elif isinstance(el, instrument.Instrument):
				out_stream.insert(instrument.Soprano())
			elif isinstance(el, (stream.Measure)):
				# select a random index for a semitone and add to outstream
				m = stream.Measure()
				for measure_el in el:
					if isinstance(el, note.Note):
						out_note = self.random_note_in_range()
						m.insert(measure_el.offset, out_note)
					else:
						m.insert(measure_el.offset, note.Rest(quarterLength=1))
				out_stream.insert(el.offset, m)
			elif isinstance (el, (note.Note, note.Rest)):
				continue
			else:
				out_stream.insert(el.offset, copy.deepcopy(el))

		return out_stream


	def show_melody_bass(self):
		"""
		Function to show melody and bassline together
		on a single system
		"""
		melody = self.melody
		bass = self.bassline

		# conjoin two parts and shows
		s = stream.Score()
		s.insert(0, stream.Part(melody))
		s.insert(0, stream.Part(bass))
		s.show()


	def meta_fitness(self, note_, index_):
		"""
		Function to combine fitness functions
		

		Parameters
		----------

			weight_dict : dictionary
				key -> name of function
				value -> importance to assessment,
					will be standardized

			note_ : note.Note
				proposed/current note
			
			index_ : int
				index of note

		Returns
		-------

			fitness score
		"""

		# standardize wieght_dict if it exists
		if self.weight_dict is not None:
			total_sum = np.sum(list(self.weight_dict.values()))
			weight_dict = {k : self.weight_dict[k] / total_sum for k in self.weight_dict.keys()}

		# to store fitness function outputs
		scores = {}
		for name, v in self.fitness_function_dict.items():
			scores[name] = v.ff(self, copy.deepcopy(note_), index_)

		# weighted average
		if self.weight_dict is not None:
			return np.nanmean([self.weight_dict[k] * scores[k] for k in scores.keys()])
		else:
			return np.nanmean(list(scores.values()))


	def random_note_in_chord(self, index_):
		"""
		Function to select a random note in the iven chord
		"""

		# select random note from chord
		if isinstance(self.chords[index_], int):
			semitones_above_tonic = self.chords[index_]
		else:
			possible_notes = self.chords[index_].split('/')
			select_idx = np.random.randint(len(possible_notes))
			semitones_above_tonic = int(possible_notes[select_idx])
		
		# return note
		tonic = self.key.getTonic()
		pitch_ = tonic.transpose(semitones_above_tonic)

		# select random obctave within vocal range
		low, high = self.vocal_range

		#lowest octave
		lowest_oct = pitch_.transposeAboveTarget(low).octave

		# highest octave
		highest_oct = pitch_.transposeBelowTarget(high).octave

		# bug sometimes arises where low higher than high
		if lowest_oct >= highest_oct:
			lowest_oct =low.octave

		# highest + 1 as numpy.random.randint is exclusive of high 
		oct_ = np.random.randint(low=lowest_oct, high=highest_oct + 1)
		pitch_.octave = oct_

		return note.Note(pitch_)

    
	def run(self, n_iter, profiling, plotting=True):
		"""
		Function to run the full MH algo

		Parameters
		----------

			n_iter : integer
				number of iterations for MH algo
		"""

		# flatten notes for quick indexing
		stream_notes = self.melody.flat.getElementsByClass([note.Note, note.Rest]).flat
		bass_notes = self.bassline.flat.getElementsByClass([note.Note, note.Rest]).flat
		stream_length = len(stream_notes)

		# create dataframe to store progress of algo
		if profiling:
			profile_df = pd.DataFrame(index=np.arange(n_iter), columns=list(self.fitness_function_dict.keys()))


		# for a given number of iterations:
		for i in np.arange(n_iter):

			# select a note/chord at random (note beginning/end difficulties)
			idx = np.random.randint(stream_length)
			curr_note = stream_notes[idx]

			# propose a new note
			prop_note = self.random_note_in_chord(idx)

			# compare current melody note with proposed note
			curr_fit = self.meta_fitness(curr_note, idx)
			prop_fit = self.meta_fitness(prop_note, idx)

			# accept or reject with a probability given by the 
			# ratio of current and proposed fitness functions
			accept_func = (prop_fit / curr_fit)
			if np.random.uniform() < accept_func:
				stream_notes[idx] = prop_note

			if profiling:
				# loop through fitness functions
				for k,v in self.fitness_function_dict.items():
					profile_df.loc[i,k] = v.profiling(bass_notes, stream_notes)

		# return to melody format
		out_stream = stream.Stream()

		# loop through measures
		note_index = 0
		for el in self.bassline.recurse(skipSelf=True):
			if el == clef.BassClef():
				out_stream.insert(clef.TrebleClef())
			elif isinstance(el, instrument.Instrument):
				out_stream.insert(instrument.Soprano())
			elif isinstance(el, (stream.Measure)):
				# select a random index for a semitone and add to outstream
				m = stream.Measure()
				for measure_el in el:
					if isinstance(el, note.Note):
						out_note = stream_notes[note_index]
						m.insert(measure_el.offset, out_note)
					else:
						m.insert(measure_el.offset, note.Rest(quarterLength=1))
					note_index += 1
				out_stream.insert(el.offset, m)
			elif isinstance (el, (note.Note, note.Rest)):
				continue
			else:
				out_stream.insert(el.offset, copy.deepcopy(el))
		
		if plotting:
			if len(profile_df.columns) == 1:
				profile_df.plot()
			else:
				# plot the results
				row_num = int(np.ceil(len(profile_df.columns) / 2))

				# initialize row and column iterators
				row = 0
				col = 0
				fig, axs = plt.subplots(row_num, 2, figsize=(12*row_num, 8))
				for i in np.arange(len(profile_df.columns)):

					ax = np.array(axs).reshape(-1)[i]
					ax.plot(profile_df.index, profile_df.iloc[:,i])
					ax.set_xlabel('N iterations')
					ax.set_ylabel(profile_df.columns[i])
					ax.set_title(profile_df.columns[i])

					# get to next axis
					if i % 2 == 1:
						col += 1
					else:
						col = 0
						row += 1

				# turn axis off if empty
				if col == 0:
					ax.axis('off')

				plt.show()

		self.melody = out_stream

		if profiling:
			print(profile_df)
			return profile_df
		else:
			return out_stream


#### FITNESS FUNCTIONS

class FitnessFunction():
	"""
	Class to store fitness functions
	"""

	def __init__(self, name):
		self.name = name

	def ff(self):
		pass

	def profiling(self):
		pass


## make fitness functions a class themselves with a profiling tag
# so that convergence can be proven or disproven


class NoLargeJumps(FitnessFunction):

	def ff(self, mh, note_, index_):
		"""
		Prefer smaller melody jumps to larger ones
		"""

		melody = mh.melody.flat.getElementsByClass([note.Note, note.Rest]).flat

		# replace original melody note with proposed note
		melody[index_] = copy.deepcopy(note_)

		jumps = np.empty((len(melody)-1,))

		# get sum of squared differences for melody
		for idx in np.arange(1, len(melody)):

			note_1_melody = melody[idx - 1]

			if note_1_melody.isRest:
				note_1_melody = data_utils.closest_note_not_rest(melody, idx, np.subtract)

			note_2_melody = melody[idx]

			if note_2_melody.isRest:
				note_2_melody = data_utils.closest_note_not_rest(melody, idx, np.add)

			if note_1_melody.isRest or note_2_melody.isRest:
				melody_jump = np.nan
			else:
				melody_jump = interval.notesToChromatic(note_1_melody, note_2_melody).semitones

			jumps[idx - 1] = melody_jump**2

		return (np.nansum(jumps) / (len(melody)-1))**(-1)


	def profiling(self, bass, melody):

		jumps = np.empty((len(melody)-1,))
		for idx in np.arange(1, len(melody)):

			note_1_melody = melody[idx - 1]

			if note_1_melody.isRest:
				note_1_melody = data_utils.closest_note_not_rest(melody, idx, np.subtract)

			note_2_melody = melody[idx]

			if note_2_melody.isRest:
				note_2_melody = data_utils.closest_note_not_rest(melody, idx, np.add)

			if note_1_melody.isRest or note_2_melody.isRest:
				melody_jump = np.nan
			else:
				melody_jump = interval.notesToChromatic(note_1_melody, note_2_melody).semitones

			jumps[idx - 1] = melody_jump**2

		return np.nansum(jumps) / (len(melody)-1)


class ContraryMotion(FitnessFunction):

	def __init__(self, name, metric):
		self.name = name
		self.metric = metric
		self.set_metric()


	def concordance_table(self, melody, bass):
		"""
		Function to product concordance table of direction
		of part movements

		Parameters
		----------

		    melody : music21.Part
		    
		    bass : music21.Part

		Returns
		-------

			concordance_table : 2x2 pandas.DataFrame
		"""

		# get flattened parts
		p1 = melody.flat.getElementsByClass([note.Note, note.Rest]).flat
		p2 = bass.flat.getElementsByClass([note.Note, note.Rest]).flat

		# set up DataFrame to produce table
		arr = np.zeros((2,2))
		idx = ['Up - Bass', 'Down - Bass']
		cols = ['Up - Melody', 'Down - Melody']
		concordance_tbl = pd.DataFrame(arr, index=idx, columns=cols)

		# loop through intervals and fill in concordance table
		for idx in np.arange(1, len(p1)):
			# directions : neg is down, pos is up

			# if rest in melody
			p1_from = melody[idx - 1]

			if p1_from.isRest:
				p1_from = data_utils.closest_note_not_rest(melody, idx, np.subtract)

			p1_to = melody[idx]

			if p1_to.isRest:
				p1_to = data_utils.closest_note_not_rest(melody, idx, np.add)

			# if rest in bass
			p2_from = bass[idx - 1]

			if p2_from.isRest:
				p2_from = data_utils.closest_note_not_rest(bass, idx, np.subtract)

			p2_to = bass[idx]

			if p2_to.isRest:
				p2_to = data_utils.closest_note_not_rest(bass, idx, np.add)

			# if begins or ends on a rest
			if p1_to.isRest or p2_to.isRest or p1_from.isRest or p2_from.isRest:
				dir_p1 = np.nan
				dir_p2 = np.nan
			else:
				dir_p1 = np.sign(interval.notesToChromatic(p1_from, p1_to).semitones)
				dir_p2 = np.sign(interval.notesToChromatic(p2_from, p2_to).semitones)

			# zero in either part should be assigned to either
			# contrary equally
			if np.isnan(dir_p1) or np.isnan(dir_p2):
				continue
			elif (dir_p1 == 0) or (dir_p2 == 0):
				rw = np.random.randint(0,2)
				if rw == 1:
					cl = 0
				else:
					cl = 1
				concordance_tbl.iloc[rw,cl] += 1
			elif (dir_p1 < 0) and (dir_p2 > 0):
			    concordance_tbl.iloc[0, 1] += 1
			elif (dir_p1 > 0) and (dir_p2 < 0):
			    concordance_tbl.iloc[1, 0] += 1
			elif (dir_p1 < 0) and (dir_p2 < 0):
			    concordance_tbl.iloc[1, 1] += 1
			elif (dir_p1 > 0) and (dir_p2 > 0):
			    concordance_tbl.iloc[0, 0] += 1

			    
		return concordance_tbl


	def set_metric(self):
		if self.metric == 'cohen_kappa':
			self.metric_func = self.cohen_kappa
		elif self.metric == 'agreement_proportion':
			self.metric_func = self.agreement_proportion
		elif self.metric == 'mcnemar':
			self.metric_func = self.mcnemar		
		else:
			print('Use one of the given metric functions')
			return -1


	def cohen_kappa(self, concordance_tbl):
		"""
		Function to perform Cohen's Kappa measure of
		contrary motiuon given a concordance table

		Paper:
		    http://journals.sagepub.com/doi/10.1177/001316446002000104

		Parameters
		----------

		    concordance_tbl : pandas.DataFrame
		        concordance table
		        
		Returns
		-------

		    kappa : float
		        Cohen's Kappa
		"""

		# number of entries total in table
		N = concordance_tbl.values.sum()

		# designate motions
		contrary = (concordance_tbl.iloc[0,0] + concordance_tbl.iloc[1,1]) / N

		# bass up at random
		p_yes = (concordance_tbl.iloc[0,0] + concordance_tbl.iloc[0,1]) * \
		    (concordance_tbl.iloc[0,0] + concordance_tbl.iloc[1,0]) / (N**2)
		p_no = (concordance_tbl.iloc[0,1] + concordance_tbl.iloc[1,1]) * \
		    (concordance_tbl.iloc[1,0] + concordance_tbl.iloc[1,1]) / (N**2)
		random_agreement = p_yes + p_no

		kappa = (contrary - random_agreement) / (1 - random_agreement)

		# to prevent negative values, add 1
		return kappa + 1


	def agreement_proportion(self, concordance_tbl):
		"""
		number contrary / total number

		Parameters
		----------

		    concordance_tbl : pandas.DataFrame
		        concordance table
		        
		Returns
		-------

		    agreement_prop : float
		"""

		# number of entries total in table
		N = concordance_tbl.values.sum()

		agreement_prop = (concordance_tbl.iloc[0,1] + concordance_tbl.iloc[1,0]) / N
		return agreement_prop

	def mcnemar(self, concordance_tbl):
		mcnemar_stat = (((concordance_tbl.iloc[1,0] - concordance_tbl.iloc[0,1])**2) / \
			(concordance_tbl.iloc[1,0] + concordance_tbl.iloc[0,1])) + 0.1
		return chi2.cdf(mcnemar_stat, df=1)
    
	def ff(self, mh, note_, index_):
		melody = mh.melody.flat.getElementsByClass([note.Note, note.Rest]).flat
		bass = mh.bassline.flat.getElementsByClass([note.Note, note.Rest]).flat

		melody[index_] = copy.deepcopy(note_)

		ct = self.concordance_table(melody, bass)
		return self.metric_func(ct)
        
	def profiling(self, bass, melody):
		melody = melody.flat.getElementsByClass([note.Note, note.Rest]).flat
		bass = bass.flat.getElementsByClass([note.Note, note.Rest]).flat

		ct = self.concordance_table(melody, bass)
		return self.metric_func(ct)
        

##### IMPLEMENTATIONS

def PachetRoySopranoAlgo(chorale):
	return MH(
		chorale,
		(pitch.Pitch('c4'), pitch.Pitch('g5')),
		{
			'NLJ' : NoLargeJumps('NLJ'),
			'CM' : ContraryMotion('CM', 'mcnemar')
		},
		weight_dict={'NLJ': 10 , 'CM' : 2}
	)


def run_batch_profile(n_iter, mh_algo, chorales):
	"""
	Function to run a batch of MH algorithms and 
	observe how the data behaves.

	Parameters
	----------

		n_iter : int
			Number of iterations to run for each chorale

		mh_algo : mh.MH
			particular mh Implementation

		chorales : list of music.Scores

	Returns
	-------

		Diagnostic information
	"""

	# length of list of chorales
	n_chorales = len(chorales)

	# prepare data structures for diagnostics
	starting_fitness = np.empty((n_chorales,))
	finishing_fitness = np.empty((n_chorales,))
	full_fitness = np.empty((n_chorales,n_iter))


	# row = 0
	# col = 0
	# fig, axs = plt.subplots(num_constr, 2, figsize=(12*num_constr, 8))
	# loop through chorales
	for i in trange(n_chorales):
		
		algo = mh_algo(chorales[i])

		# run MH algo
		meta_scores = algo.run(n_iter, True, plotting=False)

		# TODO MAKE SEPARATE FOR EACH FITNESS FUNCTION
		starting_fitness[i] = meta_scores.iloc[0,0]
		finishing_fitness[i] = meta_scores.iloc[0,-1]
		full_fitness[i, :] = meta_scores.mean(axis=1)

	# print output
	fig, ax = plt.subplots(1,1)
	for j in np.arange(n_chorales):
		ax.plot(np.arange(n_iter), full_fitness[j,:], color='blue', alpha=0.2)


