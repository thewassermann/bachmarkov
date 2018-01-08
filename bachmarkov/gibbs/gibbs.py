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
		chorale,
		soprano,
		bass,
		chords,
		vocal_range_dict,
		conditional_dict,
		weight_dict=None):

		# will remove later
		self.chorale = chorale

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

		# may need to abstract this so tha MH cant see the actual soprano line?? //TODO
		self.fermata_layer = extract_utils.extract_fermata_layer(
			extract_utils.to_crotchet_stream(self.chorale.parts['Soprano'])
		)


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
    
    
	def return_chorale(self):
		"""
		Return the chorale in its current form as a score
		"""
		# conjoin two parts and shows
		s = stream.Score()

		# add in expression
		soprano = stream.Part(self.soprano)
		out_stream = stream.Stream()
		soprano_flat = list(soprano.recurse(classFilter=('Note', 'Rest')))
		soprano = extract_utils.flattened_to_stream(
			soprano_flat,
			self.bass,
			out_stream,
			'Soprano',
			self.fermata_layer
		)
		
		soprano = stream.Part(soprano)
		soprano.id = 'Soprano'

		alto = stream.Part(self.alto)
		alto.id = 'Alto'
		tenor = stream.Part(self.tenor)
		tenor.id = 'Tenor'
		bass = stream.Part(self.bass)
		bass.id = 'Bass'

		s.insert(0, soprano)
		s.insert(0, alto)
		s.insert(0, tenor)
		s.insert(0, bass)
		return s


	def run(self, n_iter, profiling, plotting=True):
		"""
		Run the gibbs sampler for `n_iter` iterations

		Parameters
		----------

			n_iter : int
				number of iterations to run the gibbs sampler for 
		"""

		# to store profiled data
		if profiling:
			profile_arr = np.empty((n_iter, len(list(self.conditional_dict.keys()))))
			profile_df = pd.DataFrame(profile_arr)
			profile_df.index = np.arange(n_iter)
			profile_df.columns = list(self.conditional_dict.keys())

		if not plotting:
			tf_show_progress = True
		else:
			tf_show_progress = False

		for i in trange(n_iter, disable=tf_show_progress):
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
			sample_idx = np.random.choice(np.arange(len(self.soprano.recurse(classFilter=('Note', 'Rest')))))

			chosen_note = self.meta_conditional(name_, sample_line, sample_idx, sample_range)
			sample_flat = list(sample_line.recurse(classFilter=('Note', 'Rest')))
			sample_flat[sample_idx] = chosen_note


			out_stream = stream.Stream()
			if name_ == 'Alto':
				self.alto = extract_utils.flattened_to_stream(sample_flat, self.bass, out_stream, name_)
			else:
				self.tenor = extract_utils.flattened_to_stream(sample_flat, self.bass, out_stream, name_)

			if profiling:
				for k in list(self.conditional_dict.keys()):
					profile_df.loc[i, k] = self.conditional_dict[k].profiling(self)


		# display results
		if plotting:

			# create a plot for each profile_df
			N = len(profile_df.columns)
			fig, axs = plt.subplots(N, 1, figsize=(10*N, 8))

			for i in np.arange(N):
				ax = np.array(axs).flatten()[i]
				ax.plot(profile_df.index, profile_df.iloc[:, i], color='blue')
				ax.set_title(list(self.conditional_dict.keys())[i])
				ax.set_xlabel('Iterations')
				ax.set_ylabel('Profiling Score')

			fig.tight_layout()

		if profiling:
			# if more than one constraint being used
			if len(profile_df.columns) > 1:

				out_arr = np.empty((n_iter,))

				# if there is a weight dict collapse into a single 
				# weighted output
				if self.weight_dict is not None:

					for i in np.arange(n_iter):
						# get sum of weight dict so can be normalized
						weight_sum = np.nansum(list(self.weight_dict.values()))
						out_arr[i] = np.nansum(
							[profile_df.loc[i, weight_key] * self.weight_dict[weight_key] for weight_key in list(self.weight_dict.keys())]
						) / weight_sum
				# if equal weighted can just average
				else:
					out_arr = np.nanmean(profile_df, axis=1)
				out_df = pd.Series(out_arr, name='Composite Fitness')
				return out_df
			else:
				return profile_df
		else:
			return self.return_chorale()

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
			possible_octaves = set(np.arange(low_octave, high_octave+1))

			notes_with_octaves = {}
			for oct_ in possible_octaves:
				for note_ in notes:
					pitch_ = note_.pitch
					pitch_.octave = oct_
					in_note = note.Note(pitch_, quarterLength=1)
					if (in_note.pitch <= highest_note) and (in_note.pitch >= lowest_note):
						notes_with_octaves[in_note.nameWithOctave] = in_note

			# conditional probability of each possible note
			prob_dist = {}
			for possible_note in list(notes_with_octaves.keys()):

				# score for each note
				score = 0

				# loop through conditional probabilities
				for cd in list(self.conditional_dict.keys()):
					if self.weight_dict is None:
						score += self.conditional_dict[cd].dist(
							self,
							possible_note,
							part_name,
							sample_line,
							sample_idx,
							sample_range
						)
					else:
						score += self.conditional_dict[cd].dist(
							self,
							possible_note,
							part_name,
							sample_line,
							sample_idx,
							sample_range
						)

					prob_dist[possible_note] = score

			score_sum = np.nansum(list(prob_dist.values()))
			prob_dist = {k: prob_dist[k]/score_sum for k in list(prob_dist.keys())}
			# print('Soprano Note {}'.format(self.soprano.recurse(classFilter=('Note', 'Rest'))[sample_idx].nameWithOctave))
			# print('{} dist: {}'.format(part_name, prob_dist))
			# print('Sample Index {}'.format(sample_idx))
			chosen_name = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
			return note.Note(chosen_name, quarterLength=1)


class ConditionalDistribution():
	"""
	Class to store methods for calculating conditional distributions
	"""
	def __init__(self, name):
		self.name = name

	def dist(self, gs, possible_note, part_name, sample_line, sample_idx, sample_range):
		pass

	def profiling(self, gs):
		pass



class NoCrossing(ConditionalDistribution):

	def dist(self, gs, possible_note, part_name, sample_line, sample_idx, sample_range):
		"""
		No crossing between individual parts
		"""
		soprano = gs.soprano.recurse(classFilter=('Note', 'Rest'))
		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))
		bass = gs.bass.recurse(classFilter=('Note', 'Rest'))
        
		# convert to list for assignment
		lowest_note, highest_note = sample_range

		# alter sample range to fit in with the no-crossing policy
		if gs.chords[sample_idx] == -1:
			# can only be a rest
			if isinstance(possible_note, note.Rest):
				return 1.
			else:
				return .001
		else:
			if part_name == 'Alto':
				if soprano[sample_idx].pitch < highest_note:
					highest_note = soprano[sample_idx].pitch
				if tenor[sample_idx].pitch > lowest_note:
					lowest_note = tenor[sample_idx].pitch
			elif part_name == 'Tenor':
				if alto[sample_idx].pitch < highest_note:
					highest_note = alto[sample_idx].pitch
				if bass[sample_idx].pitch > lowest_note:
					lowest_note = bass[sample_idx].pitch
			    
			# if surrounding parts already crossed each note equally suggested
			if highest_note < lowest_note:
				return 1. 
			else:
				if (pitch.Pitch(possible_note) < highest_note) and \
					(pitch.Pitch(possible_note) > lowest_note):
					return 1.
				else:
					return 0.001

	def profiling(self, gs):
		"""
		See how many beats contain crossing 
		"""
		soprano = gs.soprano.recurse(classFilter=('Note', 'Rest'))
		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))
		bass = gs.bass.recurse(classFilter=('Note', 'Rest'))

		# loop through each beat and count if it contains a part 
		# crossing
		beats_no_crossing = 0

		for i in np.arange(len(soprano)):

			if soprano[i].isRest or alto[i].isRest or tenor[i].isRest or bass[i].isRest:
				continue
			elif (soprano[i].pitch > alto[i].pitch) and \
				(alto[i].pitch > tenor[i].pitch) and \
				(tenor[i].pitch > bass[i].pitch):
					beats_no_crossing += 1

		return beats_no_crossing / len(soprano)

        
class StepWiseMotion(ConditionalDistribution):

	def dist(self, gs, possible_note, part_name, sample_line, sample_idx, sample_range):
		"""
		Prefer smaller steps to larger ones
		"""
		sample_line = list(sample_line.recurse(classFilter=('Note', 'Rest')))
		sample_line[sample_idx] = copy.deepcopy(note.Note(possible_note, quarterLength=1))
        
		# if last index just look to previous note
		if sample_idx == len(sample_line) - 1:
			s = -3
			e = -1

		# if the first index just look to next note
		elif sample_idx == 0:
			s = 0
			e = 2
		else:
			s = sample_idx - 1
			e = sample_idx + 1

		intervals_particular = extract_utils.get_intervals(sample_line, s, e)
		intervals_particular = [1 if i == 0 else i for i in intervals_particular]
		intervals_particular_score = np.nansum(np.array(intervals_particular)**2)**-1.

		intervals_total = extract_utils.get_intervals(sample_line)
		intervals_total_score = np.nansum(np.array(intervals_total)**2)**-1.

		return np.average([intervals_total_score, intervals_particular_score], weights=[.05, .95])

	def profiling(self, gs):

		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))

		# extract intervals for alto and tenor parts
		alto_intervals = extract_utils.get_intervals(alto)
		tenor_intervals = extract_utils.get_intervals(tenor)

		jumps = 0
		for i in np.arange(len(alto_intervals)):

			# keep tally of jumps
			if alto_intervals[i] > 4:
				jumps += 1
			if tenor_intervals[i] > 4:
				jumps += 1

		return 1 - (jumps / (2*len(alto)))



class NoParallelMotion(ConditionalDistribution):

	def dist(self, gs, possible_note, part_name, sample_line, sample_idx, sample_range):
		"""
		Discourage (heavily) any parallel motion in 5ths or octaves
		"""

		sample_line = sample_line.recurse(classFilter=('Note', 'Rest'))
		soprano = gs.soprano.recurse(classFilter=('Note', 'Rest'))
		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))
		bass = gs.bass.recurse(classFilter=('Note', 'Rest'))

		# if first index parallel motion impossible
		if sample_idx == 0:
			return 1. 
		else:
			# make list of other parts to check
			other_parts = []
			if part_name == 'Alto':
				other_parts = [soprano, tenor, bass]
			elif part_name == 'Tenor':
				other_parts = [soprano, alto, bass]

			# loop through
			for part_ in other_parts:
				if (gs.chords[sample_idx-1] == -1) or (gs.chords[sample_idx] == -1):
					return 1.
				else:
					prev_interval = interval.notesToChromatic(sample_line[sample_idx - 1], part_[sample_idx - 1]).semitones % 12
					interval_ = interval.notesToChromatic(sample_line[sample_idx], part_[sample_idx]).semitones % 12

					if (prev_interval == interval_) and (interval in set([-5, 0, 7])):
						return 0.001

			# if at this point there is no parallel motion
			return 1.


	def profiling(self, gs):

		soprano = gs.soprano.recurse(classFilter=('Note', 'Rest'))
		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))
		bass = gs.bass.recurse(classFilter=('Note', 'Rest'))

		soprano_intervals = extract_utils.get_intervals(soprano)
		alto_intervals = extract_utils.get_intervals(alto)
		tenor_intervals = extract_utils.get_intervals(tenor)
		bass_intervals = extract_utils.get_intervals(bass)

		parallel_motions = 0
		for i in np.arange(1, len(soprano_intervals)):

			soprano_motion = (soprano_intervals[i] - soprano_intervals[i-1]) % 12
			alto_motion = (alto_intervals[i] - alto_intervals[i-1]) % 12
			tenor_motion = (tenor_intervals[i] - tenor_intervals[i-1]) % 12
			bass_motion = (bass_intervals[i] - bass_intervals[i-1]) % 12

			motions = [soprano_motion, alto_motion, tenor_motion, bass_motion]

			if (motions.count(0) > 1) or \
				(motions.count(7) > 1):
					parallel_motions += 1

		return 1 - (parallel_motions / len(soprano_intervals))


class OctaveMax(ConditionalDistribution):

	def dist(self, gs, possible_note, part_name, sample_line, sample_idx, sample_range):
		"""
		Between soprano, alto, tenor lines do not want more than an octave between each
		"""

		sample_line = sample_line.recurse(classFilter=('Note', 'Rest'))
		soprano = gs.soprano.recurse(classFilter=('Note', 'Rest'))
		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))

		# make list of other parts to check
		other_parts = []
		if part_name == 'Alto':
			other_parts = [soprano, tenor]
		elif part_name == 'Tenor':
			other_parts = [soprano, alto]

		for part_ in other_parts:
			if (gs.chords[sample_idx-1] == -1) or (gs.chords[sample_idx] == -1):
				return 1.
			else:
				interval_ = interval.notesToChromatic(sample_line[sample_idx], part_[sample_idx]).semitones
				if interval_ > 12:
					return 0.001

		# if at this point they are clustered within two octaves
		return 1.


	def profiling(self, gs):

		alto = gs.alto.recurse(classFilter=('Note', 'Rest'))
		tenor = gs.tenor.recurse(classFilter=('Note', 'Rest'))

		alto_intervals = extract_utils.get_intervals(alto)
		tenor_intervals = extract_utils.get_intervals(tenor)

		octave_jumps = 0
		for i in np.arange(len(alto_intervals)):

			if alto_intervals[i] > 12:
				octave_jumps += 1

			if tenor_intervals[i] > 12:
				octave_jumps += 1

		return 1 - (octave_jumps / (2*len(alto_intervals)))






