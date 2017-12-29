"""
File to store functions/classes for 
Metropolis-Hastings based algorithms
"""

from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats.mstats import gmean

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
		self.chorale = chorale
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
					if isinstance(measure_el, note.Note):
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


	def show_melody_bass(self, show_command=None):
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
		s.show(show_command)


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
			return gmean([self.weight_dict[k] * scores[k] for k in scores.keys()])
		else:
			return gmean(list(scores.values()))


	def random_note_in_chord(self, index_):
		"""
		Function to select a random note in the iven chord
		"""

		# select random note from chord
		if self.chords[index_] == -1:
			semitones_above_tonic = self.chords[index_]
			return note.Rest(quarterLength=1)

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
				lowest_oct = low.octave

			# highest + 1 as numpy.random.randint is exclusive of high 
			oct_ = np.random.randint(low=lowest_oct, high=highest_oct + 1)
			pitch_.octave = oct_

			return note.Note(pitch_, quarterLength=1)

    
	def run(self, n_iter, profiling, plotting=True):
		"""
		Function to run the full MH algo

		Parameters
		----------

			n_iter : integer
				number of iterations for MH algo
		"""

		# flatten notes for quick indexing
		stream_notes = list(self.melody.recurse(classFilter=('Note', 'Rest')))
		bass_notes = list(self.bassline.recurse(classFilter=('Note', 'Rest')))
		stream_length = len(stream_notes)

		# create dataframe to store progress of algo
		if profiling:
			profile_df = pd.DataFrame(index=np.arange(n_iter), columns=list(self.fitness_function_dict.keys()))

		# get positions of rests so they may be avoided
		rest_pos = []
		for i, s in enumerate(stream_notes):
			if s.isRest:
				rest_pos.append(i)

		stream_choices = [x for x in np.arange(stream_length) if x not in rest_pos]

		profiling_fits = np.empty((n_iter,))
		# for a given number of iterations:
		for i in trange(n_iter):

			# select a note/chord at random (note beginning/end difficulties)
			idx = np.random.choice(stream_choices)
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
				profiling_fits[i] = prop_fit
			else:
				profiling_fits[i] = curr_fit

			if profiling:
				# loop through fitness functions
				for k,v in self.fitness_function_dict.items():
					profile_df.loc[i,k] = v.profiling(self, bass_notes, stream_notes)

		# return to melody format
		out_stream = stream.Stream()
		out_stream = extract_utils.flattened_to_stream(stream_notes, self.bassline, out_stream, 'Soprano')
		
		if plotting:
			if len(profile_df.columns) == 1:
				profile_df.plot()
			else:
				# plot the results
				row_num = int(np.ceil((len(profile_df.columns) + 1) / 2))

				# initialize row and column iterators
				row = 0
				col = 0
				fig, axs = plt.subplots(row_num, 2, figsize=(12*row_num, 8))

				# fit plotting
				ax_fit = np.array(axs).reshape(-1)[0]
				ax_fit.plot(np.arange(n_iter), profiling_fits, color='red')
				ax_fit.set_xlabel('N iterations')
				ax_fit.set_ylabel('Fitness')
				ax_fit.set_ylim([0,10])
				ax_fit.set_title('Accepted Fitness Function')

				for i in np.arange(1, len(profile_df.columns) + 1):

					ax = np.array(axs).reshape(-1)[i]
					ax.plot(profile_df.index, profile_df.iloc[:,i-1])
					ax.set_xlabel('N iterations')
					ax.set_ylabel(profile_df.columns[i-1])
					ax.set_title(profile_df.columns[i-1])

					# get to next axis
					if i % 2 == 1:
						col += 1
					else:
						col = 0
						row += 1

				# turn axis off if empty
				if col == 0:
					np.array(axs).reshape(-1)[i].axis('off')

				fig.tight_layout()
				plt.show()

		self.melody = out_stream

		if profiling:
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

		melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))

		# replace original melody note with proposed note
		melody[index_] = copy.deepcopy(note_)

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
		return ((1/len(intervals))*(np.nansum((intervals+0.001)**2)))**-1


	def profiling(self, mh, bass, melody):

		# replace original melody note with proposed note
		intervals = extract_utils.get_intervals(melody)

		return (1/len(intervals))*(np.nansum((intervals+0.001)**2))


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

		# set up DataFrame to produce table
		arr = np.zeros((2,2))
		idx = ['Up - Bass', 'Down - Bass']
		cols = ['Up - Melody', 'Down - Melody']
		concordance_tbl = pd.DataFrame(arr, index=idx, columns=cols)

		# loop through intervals and fill in concordance table
		p1_intervals = np.empty((len(melody)-1,))
		p2_intervals = np.empty((len(melody)-1,))
		for idx in np.arange(0, len(melody)-1):
			p1_intervals[idx] = interval.notesToChromatic(melody[idx], melody[idx+1]).semitones
			p2_intervals[idx] = interval.notesToChromatic(bass[idx], bass[idx+1]).semitones

		p1_dir_arr = np.sign(p1_intervals)
		p2_dir_arr = np.sign(p2_intervals)

		for i in np.arange(0, len(melody)-1):
			dir_p1 = p1_dir_arr[i]
			dir_p2 = p2_dir_arr[i]
			# zero in either part should be assigned to either
			# contrary equally
			if (dir_p1 == 0) or (dir_p2 == 0):
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
		return kappa


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
		melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))
		bass = list(mh.bassline.recurse(classFilter=('Note', 'Rest')))

		melody[index_] = copy.deepcopy(note_)

		ct = self.concordance_table(melody, bass)
		return self.metric_func(ct)
        
	def profiling(self, mh, bass, melody):

		ct = self.concordance_table(melody, bass)
		return self.metric_func(ct)


class NotesToTonic(FitnessFunction):
	"""
	Function to prefer:
	    leading note -> tonic
	    supertonic -> tonic
	"""

	def ff(self, mh, note_, index_):

		melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))
		# if at first note nothing to be led from
		if index_ == 0:
			return 1.
		else:
			prev = melody[index_ - 1]
			if not isinstance(prev, (note.Rest)):
				tonic = mh.key.getTonic().pitchClass
				prev_pitch = prev.pitchClass
				note_pitch = note_.pitchClass
				relPitch_prev = (prev_pitch - tonic) % 12
				relPitch = (note_pitch - tonic) % 12
                
                # supertonic
				if relPitch_prev in set([1,2]):
					if relPitch == 0:
						return 1.
					else:
						return 0.001
                # leading note
				elif relPitch_prev in set([10,11]):
					if relPitch == 0:
						return 1.
					else:
						return 0.001
				else:
					return 1.
			# if rest is previous pitch
			else:
				return 1.
    
	def profiling(self, mh, bass, melody):
	        
			tonic = mh.key.getTonic().pitchClass

			# convert all to relative pitches and rests
			melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))
			relMelody = []
			for i, el in enumerate(melody):
				if not isinstance(el, (note.Rest)):
					relMelody.append((el.pitchClass - tonic) % 12)
				else:
					relMelody.append(-1)
	            
	        
	        # count total number of possible resolutions
			possible_res_count = relMelody.count(1) + \
				relMelody.count(2) + \
				relMelody.count(10) + \
				relMelody.count(11)
	            
			actual_res_count = 0
			for i in np.arange(1, len(relMelody) - 1):
				if (relMelody[i-1] in set([1,2,10,11])) and (relMelody[i] == 0):
					actual_res_count += 1
	                
			# extra added in do prevent division by zero
			return actual_res_count / (possible_res_count + 0.001)


class NoIllegalJumps(FitnessFunction):
	"""
	Function to prefer non-augmented and non-diminished jumps
	in parts
	"""

	def ff(self, mh, note_, index_):

		melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))
		# if at first note nothing to be led from
		if index_ == 0:
			return 1.
		else:
			prev = melody[index_ - 1]
			if not isinstance(prev, (note.Rest)):
				int_type_identifier = interval.Interval(prev.pitch, note_.pitch).name[0]
				if int_type_identifier not in set(['a', 'd']):
					return 1.
				else:
					return 0.001
			else:
				return 1.
    
	def profiling(self, mh, bass, melody):

		no_rests = [x for x in melody if not x.isRest]
		out_stream = []
		for i, el in enumerate(no_rests):

            # first entry has no interval
			if i == 0:
				out_stream.append('0')
			else:
				prev = no_rests[i-1]
				out_stream.append(interval.Interval(prev, el).name[0])

		return (out_stream.count('a') + out_stream.count('d')) / len(out_stream)


class NoConsecutiveIntervals(FitnessFunction):
	"""
	Function to remove consecutive intervals.

	To be used with octaves and fifths
	"""
    
	def __init__(self, name, prohibited_interval_list):
		self.name = name
		self.prohibited_interval_set = set(prohibited_interval_list)

	def ff(self, mh, note_, index_):

		# first indices cant be consecutive
		if index_ == 0 or index_ == 1:
			return 1.
		else:

			# get intervals 
			melody = list(mh.melody.recurse(classFilter=('Note', 'Rest')))
			bass = list(mh.bassline.recurse(classFilter=('Note', 'Rest')))
			melody_intervals = [i % 12 for i in extract_utils.get_intervals(melody, index_ - 2, index_)]
			bass_intervals = [i % 12 for i in extract_utils.get_intervals(bass, index_ - 2, index_)]

            # if intervals between intervals are identical
			between_intervals = (np.array(melody_intervals) - np.array(bass_intervals))
			if (between_intervals[0] == between_intervals[1]) and \
				(between_intervals[1] in self.prohibited_interval_set):
				return 0.001
			else:
				return 1.

	def profiling(self, mh, bass, melody):
		melody_intervals = [i % 12 for i in extract_utils.get_intervals(melody)]
		bass_intervals = [i % 12 for i in extract_utils.get_intervals(bass)]

		# fifths and octaves
		prohibited_interval_set = set([-5, 0, 7])

		consecutive_count = 0
		between_intervals = np.array(melody_intervals) - np.array(bass_intervals)
		for i in np.arange(1, len(melody)-1):
			if (between_intervals[i] == between_intervals[i-1]) and \
				(between_intervals[i-1] in prohibited_interval_set):
					consecutive_count += 1
		return consecutive_count / len(between_intervals)
        

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

def TsangAikenAlgo(chorale):
	return MH(
		chorale,
		(pitch.Pitch('c4'), pitch.Pitch('g5')),
		{
			'NLJ' : NoLargeJumps('NLJ'),
			'CM' : ContraryMotion('CM', 'agreement_proportion'),
			'NTT' : NotesToTonic('NTT'),
			'NIJ' : NoIllegalJumps('NIJ'),
			'NCI' : NoConsecutiveIntervals('NCI', [-5, 0, 7])
		},
		weight_dict={'NLJ': 3 , 'CM' : 10, 'NTT' : 1, 'NIJ' : 100, 'NCI' : 200}
	)


