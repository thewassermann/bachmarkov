from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats.mstats import gmean

from tqdm import trange
import matplotlib.pyplot as plt

import copy

from utils import chord_utils, extract_utils, data_utils

from mh import mh_boolean

from collections import Counter

class GibbsBooleanSampler():
	"""
	Simialr to the MCMC Boolean Sampler, designed to fill in the
	Alto and Tenor lines of the chorale

	Parameters
	----------
	
		bassline : stream.Part
			True bassline of a Bach chorale
	
		vocal_range_dict : dictionary of tuple of pitch.Pitch
			key: voice part string, value: vocal range arranged (low, high) for the part
			
		chords : list of strings
			chords with ordered pitches (e.g. `0/4/7`)

		soprano : part.Part
			output of the MH Boolean sampler -- generated melody line
	
		constraint_dict : dictionary of `Constraints`
			key : name of constraint, value : `Constraint` class

		fermata_layer : list of booleans
			1 if there is a fermata on that beat index, 0 otherwise
			
		T : numeric (float/int)
			Corresponding to `temperature`, rate at which proposals are accepted.
			This is actually the inital value of T, which is altered via self.simulated_annealing()

		alpha : float between 0 and 1
			Parameter governing cooling schedule of T
			Suggested to be between 0.85 and 0.96
			
		ps : float between 0 and 1
			Probability of performing a `metropolis_move` or a `local_search`

		weight_dict : dictionary of floats
			key : name of constraint, value : float
	"""

	def __init__(self, bassline, vocal_range_dict, chords, soprano, constraint_dict, fermata_layer, T, alpha, ps, weight_dict=None, thinning=1):

		self.bassline = extract_utils.to_crotchet_stream(bassline)
		self.key = bassline.analyze('key')
		self.chords = chords
		self.soprano = soprano
		self.constraint_dict = constraint_dict
		self.T_0 = T
		self.T = T
		self.alpha = alpha
		self.ps = ps

		self.vocal_range_dict = vocal_range_dict

		self.alto = self.init_part('Alto', self.vocal_range_dict['Alto'], self.chords)
		self.tenor = self.init_part('Tenor', self.vocal_range_dict['Tenor'], self.chords)

		self.fermata_layer = fermata_layer
		self.weight_dict = self.set_weight_dict(weight_dict)
		self.thinning = thinning


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
		for el in self.bassline.recurse(skipSelf=True):
			
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
			self.bassline,
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
		bassline = stream.Part(self.bassline)
		bassline.id = 'Bass'

		s.insert(0, soprano)
		s.insert(0, alto)
		s.insert(0, tenor)
		s.insert(0, bassline)
		return s


	def run(self, n_iter, profiling, plotting=True):
		"""
		Run the full Kitchen/Keuhlmann Algorithm
		"""

		# structure to store profiling data
		if profiling:
			# profile_array = np.empty((n_iter, len(list(self.constraint_dict.keys()))))
			# profile_df = pd.DataFrame(profile_array)
			# profile_df.index = np.arange(n_iter)
			# profile_df.columns = list(self.constraint_dict.keys())
			profile_array = np.empty((int(np.floor(n_iter/self.thinning)), ))

		for i in trange(n_iter, desc='Iteration', disable=True):
		# for i in np.arange(n_iter):

			# get length of the chorale
			alto_flat = list(self.alto.recurse(classFilter=('Note', 'Rest')))
			len_chorale = len(alto_flat)

			# get index of rests 
			no_rests_idxs = [x for x in np.arange(len_chorale) if not alto_flat[x].isRest]

			# choose a random index
			idx = np.random.choice(no_rests_idxs)

			# choose random part
			if np.random.uniform() < .5:
				target_part_name = 'Alto'
			else:
				target_part_name = 'Tenor'


			if target_part_name == 'Alto':
				self.alto = self.local_search_gibbs(self.alto, 'Alto', idx)
			else:
				self.tenor = self.local_search_gibbs(self.tenor, 'Tenor', idx)
				
			# run the profiler
			# if profiling:
			# 	for k in list(self.constraint_dict.keys()):
			# 		if target_part_name == 'Alto':
			# 			profile_df.loc[i, k] = self.constraint_dict[k].profiling(self)
			# 		else:
			# 			profile_df.loc[i, k] = self.constraint_dict[k].profiling(self)
			if profiling and (i % self.thinning == 0):
				profile_array[int(i / self.thinning)] = self.log_likelihood()

			# simulated annealing step
			self.simulated_annealing(i, self.T_0)
		
		if profiling and plotting:

			# # print out marginal chain progression
			# if len(profile_df.columns) == 1:
			# 	profile_df.plot()
			# else:
			# 	# plot the results
			# 	row_num = int(np.ceil((len(profile_df.columns)) / 2))

			# 	# initialize row and column iterators
			# 	row = 0
			# 	col = 0
			# 	fig, axs = plt.subplots(row_num, 2, figsize=(12*row_num, 8))

			# 	for i in np.arange(len(profile_df.columns)):

			# 		ax = np.array(axs).reshape(-1)[i]
			# 		ax.plot(profile_df.index, profile_df.iloc[:,i])
			# 		ax.set_xlabel('N iterations')
			# 		ax.set_ylabel(profile_df.columns[i])
			# 		ax.set_title(profile_df.columns[i])

			# 		# get to next axis
			# 		if i % 2 == 1:
			# 			col += 1
			# 		else:
			# 			col = 0
			# 			row += 1

			# 	# turn axis off if empty
			# 	if col == 0:
			# 		np.array(axs).reshape(-1)[i+1].axis('off')

			# 	fig.tight_layout()
			# 	plt.show()
			ess = self.effective_sample_size(profile_array)
			print('{} Iterations : Effective Sample Size {}'.format(n_iter, ess))

			plt.plot(np.arange(int(np.floor(n_iter/self.thinning))) * self.thinning, profile_array)

		if profiling:
			# return profile_df.mean(axis=1)
			# return the whole df
			return profile_array # profile_df


	def local_search_gibbs(self, part_, part_name, index_):
		"""
		Carry out the LocalSearch algorithm as described by Kitchen/Keuhlmann

		Search through the possible notes and select the note that does not satisfy
		the least number of constraints (a.k.a satisfies the most)
		"""
		
		possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
			extract_utils.extract_notes_from_chord(self.chords[index_]),
			self.key,
			self.vocal_range_dict[part_name]
		)
		
		# has possible notes in dict as note.Note is an unhashable type
		possible_dict = {n.nameWithOctave : n for n in possible_notes}

		# loop through possible notes and 
		# see which one satisfies the most constraints
		note_constraints_dict = {}
		for note_ in possible_notes:
			
			# create chain including possible note
			possible_chain = list(part_.recurse(classFilter=('Note', 'Rest')))
			possible_chain[index_] = note_

			if self.weight_dict is None:
				constraints_not_met = \
					[self.constraint_dict[k].not_satisfied(self, possible_chain, index_, part_name) for k in list(self.constraint_dict.keys())]
			else:
				constraints_not_met = \
					[self.constraint_dict[k].not_satisfied(self, possible_chain, index_, part_name) * \
						self.weight_dict[k] for k in list(self.constraint_dict.keys())]

			note_constraints_dict[note_.nameWithOctave] = np.nansum(constraints_not_met)
			
		# get the conditional distribution of each possible note
		# transform prob into exp(- constraints_not_met) and normalize
		note_probs = {}
		for k in list(note_constraints_dict.keys()):
			note_probs[k] = np.exp(-note_constraints_dict[k]/self.T)

		normalizing_const = np.nansum(list(note_probs.values()))
		for k in list(note_probs.keys()):
			note_probs[k] = note_probs[k] / normalizing_const

		chosen_note_name = np.random.choice(list(note_constraints_dict.keys()), p=list(note_probs.values()))
		chosen_note = note.Note(chosen_note_name)

		new_chain = list(part_.recurse(classFilter=('Note', 'Rest')))
		new_chain[index_] = chosen_note
		return extract_utils.flattened_to_stream(
			new_chain,
			self.bassline,
			stream.Stream(),
			part_name,
		)


	def log_likelihood(self):
		"""
		Compute loglikelihood of inner parts
		"""

		flat_alto = list(self.alto.recurse(classFilter=('Note', 'Rest')))
		flat_tenor = list(self.tenor.recurse(classFilter=('Note', 'Rest')))

		p_is = np.empty((len(flat_tenor * 2),))
		
		index_cnt = 0
		part_name = 'Tenor'
		
		for j in np.arange(len(flat_tenor * 2)):
			
			# index flag and previous part determine which part to operate on next
			if part_name == 'Alto':
				part_name = 'Tenor'
			else:
				part_name = 'Alto'
			
			# if rest, rest is only choice
			if self.chords[index_cnt] == -1:
				p_is[j] = 1
				continue
			
			# possible notes
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(self.chords[index_cnt]),
				self.key,
				self.vocal_range_dict[part_name]
			)
			
			# loop through notes in chorale
			note_probabilities_dict = {}
			
			# loop through possible notes alto then tenor 
			for k in np.arange(len(possible_notes)):
				
				if part_name == 'Alto':
					possible_chain = flat_alto[:]
					possible_chain[index_cnt] = possible_notes[k]
				if part_name == 'Tenor':
					possible_chain = flat_tenor[:]
					possible_chain[index_cnt] = possible_notes[k]
	
				# for each note in chorale calculate p_i
				constraints_not_met = \
					[self.constraint_dict[constraint].not_satisfied(self, possible_chain, index_cnt, part_name) * \
					self.weight_dict[constraint] for constraint in list(self.constraint_dict.keys())]
				
				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met)/self.T)
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}
			
				
			if part_name == 'Alto':
				p_is[j] = note_probabilities_dict.get(flat_alto[index_cnt].nameWithOctave, 0.01)
			else:
				p_is[j] = note_probabilities_dict.get(flat_tenor[index_cnt].nameWithOctave, 0.01)
			
			if j % 2 == 1:
				index_cnt += 1 
	
		# return scores
		return -np.nansum(np.log(p_is))

	def effective_sample_size(self, profile_array):
		rhos = profile_array[1:]
		return self.thinning * (len(profile_array) / (1 + (2 * np.nansum(rhos))))


	def simulated_annealing(self, iter_, T_0):
		"""
		Function to cool T up to a certain level
		"""
		if self.T > .1:
			self.T = self.alpha * self.T

		# lower bound to confirm convergence
		# if self.T > .01:
		#     self.T = T_0 / np.log(2 + iter_)

		# cauchy 
		# if self.T >.1:
		#     self.T = T_0 / iter_



class GibbsConstraint():
	"""
	Superclass for constraints. `Constraint`s
	return a boolean value:
		- 1 if the `Constraint` is unsatisfied
		- 0 if the `Constraint` is satisfied
	"""

	def __init__(self, name):
		self.name = name

	def not_satisfied(self):
		pass

	def profiling(self):
		pass


class NoCrossing(GibbsConstraint):
	"""
	Constraint to return 1 if the proposed note includes crossing in the
	part and 0 if it removes crossing
	"""

	def not_satisfied(self, MCMC, chain, index_, part_name):
		
		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note', 'Rest')))

		if part_name == 'Alto':
			alto_ = chain
			tenor_ = list(MCMC.tenor.recurse(classFilter=('Note', 'Rest')))
		else:
			alto_ = list(MCMC.alto.recurse(classFilter=('Note', 'Rest')))
			tenor_ = chain

		bass_ = list(MCMC.bassline.recurse(classFilter=('Note', 'Rest')))

		if self.has_crossing(
			soprano_[index_].pitch,
			alto_[index_].pitch,
			tenor_[index_].pitch,
			bass_[index_].pitch,
		) == 1:
			return 1
		else:
			return 0


	def profiling(self, MCMC):

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note')))
		alto_ = list(MCMC.alto.recurse(classFilter=('Note')))
		tenor_ = list(MCMC.tenor.recurse(classFilter=('Note')))
		bass_ = list(MCMC.bassline.recurse(classFilter=('Note')))

		crossing_count = 0
		for i in np.arange(len(soprano_)):
			if self.has_crossing(
				soprano_[i].pitch,
				alto_[i].pitch,
				tenor_[i].pitch,
				bass_[i].pitch
			) == 1:
				crossing_count += 1

		return 1 - (crossing_count/len(soprano_))


	def has_crossing(self, soprano_pitch, alto_pitch, tenor_pitch, bass_pitch):
		if (soprano_pitch > alto_pitch) and \
			(alto_pitch > tenor_pitch) and \
			(tenor_pitch > bass_pitch):
			return 0
		else:
			return 1


class StepwiseMotion(GibbsConstraint):
	"""
	Return 1 if the interval between successive notes is not a second
	"""

	def not_satisfied(self, MCMC, chain, index_, part_name):

		proposed_note = chain[index_]
		

		# if at first note -> just interval after
		if index_ == 0:

			next_note = chain[index_ + 1]

			# if next note is not a rest
			if not isinstance(next_note, (note.Rest)) and \
				(self.is_step(proposed_note, next_note) == 1):
				return 1
			else:
				return 0

		else:
			previous_note = chain[index_ - 1]

			if not isinstance(previous_note, (note.Rest)) and \
				(self.is_step(previous_note, proposed_note) == 1):
				return 1
			else:
				return 0


	def profiling(self, MCMC):

		alto_ = list(MCMC.alto.recurse(classFilter=('Note')))
		tenor_ = list(MCMC.tenor.recurse(classFilter=('Note')))

		alto_intervals = extract_utils.get_intervals(alto_)
		tenor_intervals = extract_utils.get_intervals(tenor_)

		intervals = np.append(alto_intervals, tenor_intervals)
		vals, counts = np.unique(intervals, return_counts=True)
		interval_counts = dict(zip(vals, counts))

		steps = interval_counts.get(0, 0)+ interval_counts.get(1, 0) + interval_counts.get(2, 0)

		return steps / len(intervals)


	def is_step(self, note_1, note_2):
		"""
		Function that returns 1 if `note_1` and `note_2` is within a step

		Parameters
		----------

			note_1 : note.Note
			note_2 : note.Note
		"""
		if abs(interval.notesToChromatic(note_1, note_2).semitones) >= 2:
			return 1
		else:
			return 0


class NoParallelIntervals(GibbsConstraint):
	"""
	Returns 1 if parallel intervals of octave of quint
	"""

	def not_satisfied(self, MCMC, chain, index_, part_name):

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note', 'Rest')))

		if part_name == 'Alto':
			alto_ = chain
			tenor_ = list(MCMC.tenor.recurse(classFilter=('Note', 'Rest')))
		else:
			alto_ = list(MCMC.alto.recurse(classFilter=('Note', 'Rest')))
			tenor_ = chain

		bass_ = list(MCMC.bassline.recurse(classFilter=('Note', 'Rest')))


		part_list = [soprano_, alto_, tenor_, bass_]

		# potential issues with calculation -> first note, rests
		if index_ == 0:
			return 0
		if isinstance(soprano_[index_ - 1], note.Rest) or isinstance(soprano_[index_], note.Rest):
			return 0

		for i, outer_part in enumerate(part_list):
			for j, inner_part in enumerate(part_list):

				if i < j:
					if self.is_parallel(outer_part[index_-1:index_+1], inner_part[index_-1:index_+1]) == 1:
						return 1

		# if at this point, no parallel motion
		return 0

	def profiling(self, MCMC):

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note')))
		alto_ = list(MCMC.alto.recurse(classFilter=('Note')))
		tenor_ = list(MCMC.tenor.recurse(classFilter=('Note')))
		bass_ = list(MCMC.bassline.recurse(classFilter=('Note')))


		part_list = [soprano_, alto_, tenor_, bass_]

		# loop through each note
		parallel_motions = 0

		for idx in np.arange(1, len(soprano_)):

			# only one parallel motion increment allowed per index
			index_lock = False

			for i, outer_part in enumerate(part_list):
				for j, inner_part in enumerate(part_list):

					if i < j and not index_lock:
						if self.is_parallel(outer_part[idx-1:idx+1], inner_part[idx-1:idx+1]) == 1:
							parallel_motions += 1
							index_lock = True


		return 1 - (parallel_motions / len(soprano_))

	def is_parallel(self, line_one, line_two):
		"""
		Check if there is parallel motion in octaves or fifths
		in a line of notes
		"""

		first_interval = interval.notesToChromatic(line_one[0], line_two[0]).semitones % 12
		second_interval = interval.notesToChromatic(line_one[1], line_two[1]).semitones % 12

		if (first_interval == second_interval) and (second_interval in set([0, 7])):
			return 1
		else:
			return 0


class FillInChord(GibbsConstraint):
	pass


class OctaveMax(GibbsConstraint):
	"""
	Returns 1 if soprano/alto/tenor parts more than an octave apart
	"""

	def not_satisfied(self, MCMC, chain, index_, part_name):

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note', 'Rest')))

		if isinstance(soprano_[index_], note.Rest):
			return 0


		if part_name == 'Alto':
			alto_ = chain
			tenor_ = list(MCMC.tenor.recurse(classFilter=('Note', 'Rest')))

			if (self.octave_plus(alto_[index_], soprano_[index_]) == 1) or \
				(self.octave_plus(tenor_[index_], alto_[index_]) == 1):
				return 1
			else:
				return 0
		else:
			alto_ = list(MCMC.alto.recurse(classFilter=('Note', 'Rest')))
			tenor_ = chain

			if self.octave_plus(tenor_[index_], alto_[index_]) == 1:
				return 1
			else:
				return 0


	def profiling(self, MCMC):

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note')))
		alto_ = list(MCMC.alto.recurse(classFilter=('Note')))
		tenor_ = list(MCMC.tenor.recurse(classFilter=('Note')))

		more_than_octave = 0
		for i in np.arange(len(soprano_)):

			if (self.octave_plus(alto_[i], soprano_[i]) == 1) or \
				(self.octave_plus(tenor_[i], alto_[i]) == 1):
				more_than_octave += 1

		return 1 - (more_than_octave / len(soprano_))


	def octave_plus(self, note_1, note_2):

		if abs(interval.notesToChromatic(note_1, note_2).semitones) > 12:
			return 1
		else:
			return 0


class NewNotes(GibbsConstraint):
	"""
	Constraint to return 1 if the proposed note is already present in the
	music
	"""

	def not_satisfied(self, MCMC, chain, index_, part_name):
		
		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note', 'Rest')))
		bass_ = list(MCMC.bassline.recurse(classFilter=('Note', 'Rest')))

		if part_name == 'Alto':
			tenor_ = list(MCMC.tenor.recurse(classFilter=('Note', 'Rest')))
			other_notes = [soprano_[index_], tenor_[index_], bass_[index_]]
		else:
			alto_ = list(MCMC.alto.recurse(classFilter=('Note', 'Rest')))
			other_notes = [soprano_[index_], alto_[index_], bass_[index_]]

		if self.not_new_note(chain[index_], other_notes) == 1:
			return 1
		else:
			return 0


	def profiling(self, MCMC):
		"""
		One note will have to be doubled so make sure any single note does not have more
		than 2 and that only 1 note has two
		"""

		soprano_ = list(MCMC.soprano.recurse(classFilter=('Note')))
		alto_ = list(MCMC.alto.recurse(classFilter=('Note')))
		tenor_ = list(MCMC.tenor.recurse(classFilter=('Note')))
		bass_ = list(MCMC.bassline.recurse(classFilter=('Note')))

		double_count = 0

		for i in np.arange(len(soprano_)):
			notes_in_chord = [soprano_[i].pitch.name, alto_[i].pitch.name, tenor_[i].pitch.name, bass_[i].pitch.name]
			notes_counter = Counter(notes_in_chord)
			if any([True if nc > 2 else False for nc in list(notes_counter.values())]) or (list(notes_counter.values()).count(2) > 1):
				double_count += 1

		return 1 - (double_count/len(soprano_))


	def not_new_note(self, prop_note, other_notes):
		if prop_note.pitch.name in set([on.name for on in other_notes]):
			return 1
		else:
			return 0


class CrossConstraint(GibbsConstraint):

	def __init__(self, name, constraint_list):
		self.name = name
		self.cl = constraint_list

	def not_satisfied(self, MCMC, chain, index_, part_name):
		return np.nanprod([c.not_satisfied(MCMC, chain, index_, part_name) for c in self.cl])

	def profiling(self, MCMC):
		return np.nanprod([c.profiling(MCMC) for c in self.cl])




