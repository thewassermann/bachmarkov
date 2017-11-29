"""
File to store functions/classes for 
Metropolis-Hastings based algorithms
"""

from  music21 import *
import numpy as np

from tqdm import trange

from utils import chord_utils, extract_utils

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

		low_note = note.Note(low.nameWithOctave)

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

		# TODO, maybe make diatonic for easier convergence?

		# loop through measures
		for el in self.bassline.iter:
			if el.isNote:
				if not el.isRest:
					# select a random index for a semitone and add to outstream
					out_note = self.random_note_in_range()
					out_stream.append(out_note)
			else:
				out_stream.append(el)

		return out_stream


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
		for name, ff in self.fitness_function_dict.items():
			scores[name] = ff(self, note_, index_)

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

    
	def run(self, n_iter):
		"""
		Function to run the full MH algo

		Parameters
		----------

			n_iter : integer
				number of iterations for MH algo
		"""

		# flatten notes for quick indexing
		stream_notes = self.melody.flat.getElementsByClass([note.Note, note.Rest]).flat
		stream_length = len(stream_notes)

		# for a given number of iterations:
		for i in trange(n_iter):

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
			accept_func = prop_fit / curr_fit

			if np.random.uniform() <= accept_func:
				stream_notes[idx] = prop_note


		# return to melody format
		out_stream = stream.Stream()

		# loop through measures
		for i, el in enumerate(self.melody.iter):
			if el.isNote:
				out_note = stream_notes[i]
				out_stream.append(out_note)
			else:
				out_stream.append(el)
		    
		return out_stream


#### FITNESS FUNCTIONS

def ContraryMotion(self, note_, index_):
	"""
	Contrary motion fitness function
	""" 
	bass = self.bassline.flat.getElementsByClass([note.Note, note.Rest]).flat
	melody = self.melody.flat.getElementsByClass([note.Note, note.Rest]).flat

	# if first index go forward
	if index_ == 0:
		note_1_melody = note_
		note_2_melody = melody[1]

		note_1_bass = bass[0]
		note_2_bass = bass[1]
	else:
	# else go backward

		note_1_melody = melody[index_ -1]
		note_2_melody = note_      

		note_1_bass = bass[index_ - 1]
		note_2_bass = bass[index_]

	melody_direction = np.sign(interval.notesToChromatic(note_1_melody, note_2_melody).semitones)
	bass_direction = np.sign(interval.notesToChromatic(note_1_bass, note_2_bass).semitones)

	# if contray motion : note, ratio given by pachet and roy analysis [EDA]
	if melody_direction * bass_direction <= 0:
		return 0.7078
	# else if parallel motion
	else:
		return (1 - 0.7078)


def NoLargeJumps(self, note_, index_):
	"""
	Prefer smaller melody jumps to larger ones

	Currently linear by semitones
	"""

	low, high = self.vocal_range
	semitones_in_range = interval.notesToChromatic(low, high).semitones

	melody = self.melody.flat.getElementsByClass([note.Note, note.Rest]).flat

	# if first index go forward
	if index_ == 0:
		note_1_melody = note_
		note_2_melody = melody[1]
	else:
	# else go backward

		note_1_melody = melody[index_ -1]
		note_2_melody = note_      

	melody_jump = interval.notesToChromatic(note_1_melody, note_2_melody).semitones
	jump_to_range_ratio = melody_jump / semitones_in_range

	return 1 - jump_to_range_ratio
        

##### IMPLEMENTATIONS

def PachetRoySopranoAlgo(chorale):
	return MH(
		chorale,
		(pitch.Pitch('c4'), pitch.Pitch('g5')),
		{
			'NLJ' : NoLargeJumps,
			'CM' : ContraryMotion
		},
		#weight_dict={'NLJ' : 1, 'CM' : 3}
	)
        