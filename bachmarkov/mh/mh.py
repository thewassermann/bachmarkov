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

	def __init__(self, chorale, vocal_range):
		self.bassline = extract_utils.to_crotchet_stream(chorale.parts['Bass'])
		self.chords = chord_utils.degrees_on_beats(chorale) 
		self.vocal_range = vocal_range
		self.key = chorale.analyze('key')

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
        
	def fitness(self, note_, index_):
		"""
		Function to calculate fitness function of 
		melody with chords and bassline
		"""

		pass
    
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
		    prop_note = self.random_note_in_range()

		    # compare current melody note with proposed note
		    curr_fit = self.fitness(curr_note, idx)
		    prop_fit = self.fitness(prop_note, idx)
		    
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


class MHDiatonic(MH):
    """
    Most basic: pick notes in chord
    """
    
    def fitness(self, note_, index_):

        # tonic of key
        tonic = self.key.getTonic().pitchClass
        
        # pitch 
        pitch =  note_.pitchClass
        
        # note transform
        rel_pitch = (pitch - tonic) % 12
        
        # notes in chord
        possible_notes = [int(x) for x in self.chords[index_].split('/')]
        
        if rel_pitch not in possible_notes:
            return 0.0001
        else:
            return 1
        
        