from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2

from tqdm import trange
import matplotlib.pyplot as plt

import importlib
import copy

from utils import chord_utils, extract_utils, data_utils


class GibbsSampler():
	"""
	Class to perform gibbs sampling proceedure to 
	generate alto and tenor parts
	"""

	def __init__(
		self,
		soprano,
		bass,
		chords,
		vocal_range_dict,
		conditional_dict,
		weight_dict=None):

		self.soprano = soprano
		self.bass = bass
		self.chords = chords
		self.alto_range = vocal_range_dict['Alto']
		self.tenor_range = vocal_range_dict['Tenor']
		self.conditional_dict = conditional_dict
		self.weight_dict = self.set_weight_dict(weight_dict)

		self.key = bass.analyze('key')
		self.alto = self.init_part('Alto', self.alto_range, self.chords)
		self.tenor = self.init_part('Tenor', self.tenor_range, self.chords)


	def set_weight_dict(self, weight_dict):
		"""
		Function to normalize the weight dict so that the entries 
		retain proportion but sum to 1
		"""
		if weight_dict is None:
			return None
		else:
			wd_sum = np.nansum(list(weight_dict.values()))
			return {k: weight_dict[k]/wd_sum for k in list(weight_dict.keys())}
        
        
	def init_part(self, part_name, part_range, chords):
		"""
		Function to produce a preliminary part
		"""

		out_stream = stream.Stream()

		# get correct clef for part
		if part_name == 'Tenor':
			clef_ = clef.Treble8vbClef()
		else:
			clef_ = clef.TrebleClef()
            
		# get correct part name
		if part_name == 'Tenor':
			name_ = instrument.Tenor()
		else:
			name_ = instrument.Alto()
		    
		# initialize index for chord list
		chord_idx = 0
        
		# need to get time signature, key signature etc, from bass part
		for el in self.bass.recurse(skipSelf=True):
            
			# set clef
			if el == clef.BassClef():
				out_stream.insert(clef_)
			# set part
			elif isinstance(el, instrument.Instrument):
				out_stream.insert(name_)
                
			elif isinstance(el, (stream.Measure)):
				# select a random index for a semitone and add to outstream
				m = stream.Measure()
				for measure_el in el:
					if isinstance(measure_el, note.Note):
                        
						# get random note in chord in range
						out_note_choices =  chord_utils.random_note_in_chord_and_vocal_range(
							extract_utils.extract_notes_from_chord(self.chords[chord_idx]),
							self.key,
							part_range)
						out_note = np.random.choice(out_note_choices)
						m.insert(measure_el.offset, out_note)
						chord_idx += 1
					else:
						m.insert(measure_el.offset, note.Rest(quarterLength=1))
						chord_idx += 1
				out_stream.insert(el.offset, m)
			elif isinstance (el, (note.Note, note.Rest)):
				continue
			else:
				out_stream.insert(el.offset, copy.deepcopy(el))
		return out_stream
    
    
	def show_chorale(self):
		"""
		Print the chorale in its current form 
		"""
		# conjoin two parts and shows
		s = stream.Score()
		s.insert(0, stream.Part(self.soprano))
		s.insert(0, stream.Part(self.alto))
		s.insert(0, stream.Part(self.tenor))
		s.insert(0, stream.Part(self.bass))
		s.show()


	def run(self, n_iter):
		"""
		Run the gibbs sampler for `n_iter` iterations

		Parameters
		----------

			n_iter : int
				number of iterations to run the gibbs sampler for 
		"""

		for i in trange(n_iter):
			# flip a coin for alto or tenor
			if np.random.uniform() < .5:
				# alto
				sample_line = self.alto
				sample_range = self.alto_range
				name_ = 'Alto'
			else:
				# tenor
				sample_line = self.tenor
				sample_range = self.tenor_range
				name_ = 'Tenor'

			# get index for conditional distribution
			sample_idx = np.random.choice(np.arange(len(self.soprano)))

			chosen_note = self.meta_conditional(name_, sample_line, sample_idx, sample_range)
			sample_flat = list(sample_line.recurse(classFilter=('Note', 'Rest')))
			sample_flat[sample_idx] = chosen_note


			out_stream = stream.Stream()
			if name_ == 'Alto':
				self.alto = extract_utils.flattened_to_stream(sample_flat, self.bass, out_stream, name_)
			else:
				self.tenor = extract_utils.flattened_to_stream(sample_flat, self.bass, out_stream, name_)

		# display results
		self.show_chorale()

	def meta_conditional(self, part_name, sample_line, sample_idx, sample_range):

		# first, get a list of possible notes
		lowest_note, highest_note = sample_range

		relPitchList = extract_utils.extract_notes_from_chord(self.chords[sample_idx])

		# a rest can only produce another rest
		if relPitchList == -1:
			return note.Rest(quarterLength=1)

		else:
			# turn relative Pitch into a note.Note
			notes = [] 
			for relPitch in relPitchList:
				if relPitch != 0:
					interval_ = interval.ChromaticInterval(relPitch)
					notes.append(note.Note(interval_.transposePitch(self.key.getTonic()), quarterLength=1))
				else:
					# needed as no transposition by 0 possible
					notes.append(note.Note(self.key.getTonic(), quarterLength=1))

			# these notes in all available octaves
			# only possible octaves are the ones in the range
			low_octave = lowest_note.octave
			high_octave = highest_note.octave
			possible_octaves = set([low_octave, high_octave])

			notes_with_octaves = {}
			for oct_ in possible_octaves:
				for note_ in notes:
					in_note = note.Note(note_.pitch, quarterLength=1)
					in_note.octave = oct_
					notes_with_octaves[in_note.nameWithOctave] = in_note

			# conditional probability of each possible note
			prob_dist = {}
			for possible_note in list(notes_with_octaves.keys()):

				# score for each note
				score = 0

				# loop through conditional probabilities
				for cd in list(self.conditional_dict.keys()):
					if self.weight_dict is None:
						score += self.conditional_dict[cd].dist(self, possible_note)
					else:
						score += self.conditional_dict[cd].dist(self, possible_note) * weight_dict[cd]
					prob_dist[possible_note] = score

			score_sum = np.nansum(list(prob_dist.values()))
			prob_dist = {k: prob_dist[k]/score_sum for k in list(prob_dist.keys())}
			chosen_name = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
			return notes_with_octaves[chosen_name]


class ConditionalDistribution():
	"""
	Class to store methods for calculating conditional distributions
	"""
	def __init__(self, name):
		self.name = name

	def dist(self, possible_note):
		pass


class TestDist(ConditionalDistribution):
	def dist(self, gs, possible_note):
		return 1