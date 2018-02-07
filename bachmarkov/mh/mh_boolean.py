from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats.mstats import gmean

from tqdm import trange
import matplotlib.pyplot as plt

import copy

from utils import chord_utils, extract_utils, data_utils

from statsmodels.tsa.stattools import acf

from gibbs import gibbs_boolean


class MCMCBooleanSampler():
	"""
	Implementation of the MCMC Boolean Sampler proposed by 
	`An MCMC Sampler for Mixed Boolean/Integer Constraints`
	by Kitchen and Keuhlmann
	
	Parameters
	----------
	
		bassline : stream.Part
			True bassline of a Bach chorale
	
		vocal_range : tuple of pitch.Pitch
			vocal range arranged (low, high) for the part
			
		chords : list of strings
			chords with ordered pitches (e.g. `0/4/7`)
	
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

		necessary_list : list of str
			names of constraints that should profile to 100% 
			before stopping
	"""
	
	def __init__(self, bassline, vocal_range, chords, constraint_dict, fermata_layer, T, alpha, ps, weight_dict=None, thinning=1, progress_bar_off=True, cross_constraints=False):
		
		self.bassline = extract_utils.to_crotchet_stream(bassline)
		self.bass = list(self.bassline.recurse(classFilter=('Note', 'Rest')))
		self.key = bassline.analyze('key')
		self.vocal_range = vocal_range
		self.chords = chords
		self.constraint_dict = constraint_dict
		self.T_0 = T
		self.T = T
		self.alpha = alpha
		self.ps = ps
		self.soprano = self.init_melody()
		self.melody = list(self.soprano.recurse(classFilter=('Note', 'Rest')))
		self.fermata_layer = fermata_layer
		self.weight_dict = self.set_weight_dict(weight_dict)
		self.thinning = thinning
		self.progress_bar_off = progress_bar_off
		self.cross_constraints = cross_constraints


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
		
		
	def init_melody(self):
		"""
		Function to initialize melody line with 
		random sequence of notes within a vocal range
		
		Returns
		-------

			out_stream : music21.stream
				part initialized to random note in the range

		"""
		out_stream = stream.Stream()

		# initialize index for chord list
		chord_idx = 0

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

						out_note_choices =  chord_utils.random_note_in_chord_and_vocal_range(
							extract_utils.extract_notes_from_chord(self.chords[chord_idx]),
							self.key,
							self.vocal_range,
							None,
						)
													
						out_note = np.random.choice(out_note_choices)
						m.insert(measure_el.offset, out_note)
					else:
						m.insert(measure_el.offset, note.Rest(quarterLength=1))
					chord_idx += 1
				out_stream.insert(el.offset, m)
			elif isinstance (el, (note.Note, note.Rest)):
				continue
			else:
				# may need deepcopy
				out_stream.insert(el.offset, copy.copy(el))

		return out_stream
		
	
	def run(self, n_iter, profiling, plotting=True):
		"""
		Run the full Kitchen/Keuhlmann Algorithm
		"""
		
		# get length of the chorale
		flat_chorale = list(self.soprano.recurse(classFilter=('Note', 'Rest')))
		len_chorale = len(flat_chorale)
		
		# get index of rests 
		no_rests_idxs = [x for x in np.arange(len_chorale) if not flat_chorale[x].isRest]
		
		# structure to store profiling data
		if profiling:
			# profile_array = np.empty((n_iter, len(list(self.constraint_dict.keys()))))
			# profile_df = pd.DataFrame(profile_array)
			# profile_df.index = np.arange(n_iter)
			# profile_df.columns = list(self.constraint_dict.keys())
			profile_array = np.empty((int(np.floor(n_iter/self.thinning)), ))


		loop_count = 0
		for i in trange(n_iter, desc='Iteration', disable=self.progress_bar_off):
		# for i in np.arange(n_iter):
			
			# loop through forwards and bachwards
			idx = no_rests_idxs[i - (loop_count * len(no_rests_idxs))]
			
			# choose whether to local search or metropolis
			if np.random.uniform() <= self.ps:
				self.soprano = self.local_search(self.soprano, idx)

			else:

				if idx == 0:
					# propose a note
					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[idx]),
						self.key,
						self.vocal_range,
						None
					)

				else:

					possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
						extract_utils.extract_notes_from_chord(self.chords[idx]),
						self.key,
						self.vocal_range,
						flat_chorale[idx-1]
					)

				proposed_note = np.random.choice(possible_notes)
				self.soprano = self.metropolis_move(self.soprano, proposed_note, idx)
				
				
			# run the profiler
			# for k in list(self.constraint_dict.keys()):
				# profile_df.loc[i, k] = self.constraint_dict[k].profiling(self, self.soprano)
			if profiling and (i % self.thinning == 0):
				profile_array[int(i / self.thinning)] = self.log_likelihood()

			# # simulated annealing step
			# self.simulated_annealing(i, self.T)

			# decide whether to loop or not
			if i % len(no_rests_idxs) == len(no_rests_idxs) - 1:
				loop_count += 1

			# backwards on alternate loops
			if loop_count % 2 == 1:
				no_rests_idxs.reverse()


				
		#self.show_melody_bass()
		
		if profiling and plotting:

			# # print out marginal chain progression
			# if len(profile_df.columns) == 1:
			# 	profile_df.plot()
			# else:
			# 	# plot the results
			# 	row_num = int(np.ceil((len(profile_df.columns) + 1) / 2))

			# 	# initialize row and column iterators
			# 	row = 0
			# 	col = 0
			# 	fig, axs = plt.subplots(row_num, 2, figsize=(12*row_num, 8))

			# 	ax0 = np.array(axs).reshape(-1)[0]
			# 	ax0.plot(profile_df.index, np.matmul(profile_df, (list(self.weight_dict.values()))), color='red')
			# 	ax0.set_title('Weighted Average')
			# 	ax0.set_xlabel('N iterations')

			# 	for i in np.arange(len(profile_df.columns)):

			# 		ax = np.array(axs).reshape(-1)[i + 1]
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
			# 	if col == 1:
			# 		np.array(axs).reshape(-1)[i+2].axis('off')

			# 	fig.tight_layout()



				# plt.show()

			# return the whole df
			# return profile_df
			ess = self.effective_sample_size(profile_array)
			print('{} Iterations : Effective Sample Size {}'.format(n_iter, ess))

			plt.plot(np.arange(int(np.floor(n_iter/self.thinning))) * self.thinning, profile_array)

		if profiling:
			# return profile_df.mean(axis=1)
			# return the whole df
			return profile_array # profile_df

		
	def show_melody_bass(self, show_command=None):
		"""
		Function to show melody and bassline together
		on a single system
		"""
		melody = self.soprano
		bass = self.bassline

		# conjoin two parts and shows
		s = stream.Score()

		out_stream = stream.Stream()
		soprano_flat = list(melody.recurse(classFilter=('Note', 'Rest')))
		melody = extract_utils.flattened_to_stream(
			soprano_flat,
			self.bassline,
			out_stream,
			'Soprano',
			self.fermata_layer
		)

		s.insert(0, stream.Part(melody))
		s.insert(0, stream.Part(bass))
		s.show(show_command)


	def effective_sample_size(self, profile_array):
		rhos = profile_array[1:]
		return self.thinning * (len(profile_array) / (1 + (2 * np.nansum(rhos))))


	def log_likelihood(self):
		"""
		Calculate the loglikelihood of the chain for each iteration
		"""

		soprano = list(self.soprano.recurse(classFilter=('Note', 'Rest')))
		bass = list(self.bassline.recurse(classFilter=('Note', 'Rest')))

		p_is = np.empty((len(bass), ))

		for j in np.arange(len(bass)):
			
			# if rest, rest is only choice
			if self.chords[j] == -1 or isinstance(soprano[j], note.Rest) or isinstance(bass[j], note.Rest):
				p_is[j] = 1
				continue
			
			if j == 0:

				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					None,
				)

			else:

				# possible notes
				possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
					extract_utils.extract_notes_from_chord(self.chords[j]),
					self.key,
					self.vocal_range,
					soprano[j-1],
				)
			
			# loop through notes in chorale
			note_probabilities_dict = {}
			
			# loop through possible notes
			for k in np.arange(len(possible_notes)):
				
				possible_chain = soprano[:]
				possible_chain[j] = possible_notes[k]
	
				# for each note in chorale calculate p_i
				constraints_not_met = {constraint : self.constraint_dict[constraint].not_satisfied(self, possible_chain, j) for constraint in list(self.constraint_dict.keys())}

				# add cross constraints if necessary
				if self.cross_constraints:
					for i_, k in enumerate(list(self.constraint_dict.keys())):
						for j_, l in enumerate(list(self.constraint_dict.keys())):
							if i_ < j_:
								constraints_not_met['{}/{}'.format(k,l)] = constraints_not_met[k] * constraints_not_met[l]

				constraints_not_met = \
					[constraints_not_met[cnm] * self.weight_dict[cnm] for cnm in list(constraints_not_met.keys())]
				
				note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met))
				total_sum = np.nansum(list(note_probabilities_dict.values()))
				note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}
			
			p_is[j] = note_probabilities_dict.get(soprano[j].nameWithOctave, 0.01)

		return -np.nansum(np.log(p_is))
		

	def metropolis_move(self, part_, proposed_note, index_):
		"""
		Carry out the MetropolisMove algorithm as described by Kitchen/Keuhlmann
		
		Compare the chain with the proposed and current note. 
		Then accept the proposed note with a particular acceptance probability 
		"""
		
		# turn part_ into a list
		current_chain = list(part_.recurse(classFilter=('Note', 'Rest')))
		
		# create the proposal chain
		proposed_chain = current_chain[:]
		proposed_chain[index_] = proposed_note
		
		# get number of constraints unsatisfied with current and proposed chains
		if self.weight_dict is None:
			proposed_constraints = \
				[self.constraint_dict[k].not_satisfied(self, proposed_chain, index_) for k in list(self.constraint_dict.keys())]
			current_constraints = \
				[self.constraint_dict[k].not_satisfied(self, current_chain, index_) for k in list(self.constraint_dict.keys())]
		else:
			proposed_constraints = \
				[self.constraint_dict[k].not_satisfied(self, proposed_chain, index_) * \
					self.weight_dict[k] for k in list(self.constraint_dict.keys())]
			current_constraints = \
				[self.constraint_dict[k].not_satisfied(self, current_chain, index_) * \
					self.weight_dict[k] for k in list(self.constraint_dict.keys())]      
			
		proposed_constaints_count = np.nansum(proposed_constraints)
		current_constaints_count = np.nansum(current_constraints)
		
		# acceptance probability
		p = np.nanmin(
			[
				1,
				np.exp(-(proposed_constaints_count - current_constaints_count)/ self.T)
			]
		)
		
		# accept with probability p
		if np.random.uniform() <= p:
			return extract_utils.flattened_to_stream(
				proposed_chain,
				self.bassline,
				stream.Stream(),
				'Soprano'
			)
		else:
			return part_


	def stop_conditions(self):
		"""
		Define conditions for stopping the algorithm
		"""
	
	def local_search(self, part_, index_):
		"""
		Carry out the LocalSearch algorithm as described by Kitchen/Keuhlmann
		
		Search through the possible notes and select the note that does not satisfy
		the least number of constraints (a.k.a satisfies the most)
		"""

		possible_chain = list(part_.recurse(classFilter=('Note', 'Rest')))

		if index_ == 0:
		
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(self.chords[index_]),
				self.key,
				self.vocal_range,
				None,
			)

		else:	
		
			possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
				extract_utils.extract_notes_from_chord(self.chords[index_]),
				self.key,
				self.vocal_range,
				possible_chain[index_-1],
			)

		# has possible notes in dict as note.Note is an unhashable type
		possible_dict = {n.nameWithOctave : n for n in possible_notes}
		
		# loop through possible notes and 
		# see which one satisfies the most constraints
		note_constraints_dict = {}
		for note_ in possible_notes:
			
			# create chain including possible note
			possible_chain[index_] = note_
			
			if self.weight_dict is None:
				constraints_not_met = \
					[self.constraint_dict[k].not_satisfied(self, possible_chain, index_) for k in list(self.constraint_dict.keys())]
			else:
				constraints_not_met = \
					[self.constraint_dict[k].not_satisfied(self, possible_chain, index_) * \
						self.weight_dict[k] for k in list(self.constraint_dict.keys())]

			note_constraints_dict[note_.nameWithOctave] = np.nansum(constraints_not_met)
			
		# get best possible note from possible notes
		best_note = possible_dict[min(note_constraints_dict, key=note_constraints_dict.get)]
		
		# if a tie between best notes
		# TODO
		
		new_chain = list(part_.recurse(classFilter=('Note', 'Rest')))
		new_chain[index_] = best_note
		return extract_utils.flattened_to_stream(
			new_chain,
			self.bassline,
			stream.Stream(),
			'Soprano'
		)


	def simulated_annealing(self, iter_, T_0):
		"""
		Function to cool T up to a certain level
		"""
		if self.T > 1:
			self.T = self.alpha * self.T

		#lower bound to confirm convergence
		# if self.T > 1:
		# 	self.T = T_0 / np.log(2 + iter_)

		# cauchy 
		# if self.T >.1:
		#     self.T = T_0 / iter_


		
		
class Constraint():
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


class NoIllegalJumps(Constraint):
	"""
	Constraint function to discourage illegal jumps (aug/dim intervals)
	before and after the note
	"""
	
	def not_satisfied(self, MCMC, chain, index_):
		
		proposed_note = chain[index_]

		# if at first note -> just interval after
		if index_ == 0:
			
			next_note = chain[index_ + 1]
			
			# if next note is not a rest
			if not isinstance(next_note, (note.Rest)) and \
				(self.is_illegal(proposed_note, next_note) == 1):
				return 1
			else:
				return 0
			
		# if at last note -> just interval before
		elif index_ == len(chain) - 1:
			
			previous_note = chain[index_ - 1]
			
			# if previous note is not a rest
			if not isinstance(previous_note, (note.Rest)) and \
				(self.is_illegal(previous_note, proposed_note) == 1):
				return 1
			else:
				return 0
			
		else:
			
			previous_note = chain[index_ - 1]
			next_note = chain[index_ + 1]
			
			# assuming neither 
			if (not isinstance(previous_note, (note.Rest)) and \
				(self.is_illegal(previous_note, proposed_note) == 1)) or \
				(not isinstance(next_note, (note.Rest)) and \
				(self.is_illegal(proposed_note, next_note) == 1)):
				return 1
			else:
				return 0
			
	def profiling(self, MCMC, line):
		"""
		Profile the marginal progress of this constraint
		"""
		
		line = list(line.recurse(classFilter=('Note')))
		
		out_stream = []
		for i in np.arange(len(line)):
			out_stream.append(self.not_satisfied(MCMC, line, i))
			
		return 1 - (np.nansum(out_stream) / len(out_stream))
			
			
	def is_illegal(self, note_1, note_2):
		"""
		Helper function to see if the interval between two notes is augmented or diminished
		
		Parameters
		----------
		
			note_1 : note.Note
			note_2 : note.Note
			
		Returns
		-------
		
			bool
		"""
		
		# if augmented or diminished interval
		if (interval.notesToChromatic(note_1, note_2).semitones  % 12) == 6:
			return 1
		else:
			return 0
		
		

class NoParallelIntervals(Constraint):
	"""
	Ensure that there are no parallel octaves and fifths
	"""
	
	def not_satisfied(self, MCMC, chain, index_):
		
		bass = MCMC.bass
		
		proposed_note = chain[index_] 
		
		# if at first note -> just intervals during and after
		if index_ == 0:
			
			# get the notes needed
			proposed_note_bass = bass[index_]
			next_note_melody = chain[index_ + 1]
			next_note_bass = bass[index_ + 1]
			
			bass_ = [proposed_note_bass, next_note_bass]
			melody_ = [proposed_note, next_note_melody]
			
			# if current and next note is not a rest
			if not isinstance(next_note_bass, (note.Rest)) and \
				(self.is_parallel(bass_, melody_) == 1):
				return 1
			else:
				return 0
			
		# if at last note -> just interval before
		elif index_ == len(chain) - 1:
			
			# get the notes needed
			previous_note_melody = chain[index_ - 1]
			previous_note_bass = bass[index_ - 1]
			proposed_note_bass = bass[index_]
			
			bass_ = [previous_note_bass, proposed_note_bass]
			melody_ = [previous_note_melody, proposed_note]
						
			# if previous note is not a rest
			if not isinstance(previous_note_bass, (note.Rest)) and \
				(self.is_parallel(bass_, melody_) == 1):
				return 1
			else:
				return 0
			
		else:
			
			# get notes needed
			next_note_melody = chain[index_ + 1]
			next_note_bass = bass[index_ + 1]
			previous_note_melody = chain[index_ - 1]
			previous_note_bass = bass[index_ - 1]
			proposed_note_bass = bass[index_]
			
			first_bass = [previous_note_bass, proposed_note_bass]
			second_bass = [proposed_note_bass, next_note_bass]
			first_melody = [previous_note_melody, proposed_note]
			second_melody = [proposed_note, next_note_melody]

			notes = [next_note_melody, next_note_bass, previous_note_melody, previous_note_bass, proposed_note, proposed_note_bass]
			
			# assuming neither 
			if  not any([isinstance(n, note.Rest) for n in notes]) and \
				((self.is_parallel(first_bass, first_melody) == 1) or \
				(self.is_parallel(second_bass, second_melody) == 1)):
				return 1
			else:
				return 0
			
	def profiling(self, MCMC, line):
		
		bass = list(MCMC.bassline.recurse(classFilter=('Note')))
		melody = list(line.recurse(classFilter=('Note')))
		
		parallel_movements = 0
		for i in np.arange(1, len(melody)):
			
			prev_melody = melody[i - 1]
			prev_bass = bass[i - 1]
			
			curr_melody = melody[i]
			curr_bass = bass[i]
			
			melody_ = [prev_melody, curr_melody]
			bass_ = [prev_bass, curr_bass]
			
			if self.is_parallel(bass_, melody_) == 1:
				parallel_movements += 1
			
		return 1 - (parallel_movements / (len(melody) - 1))
		
	
	def is_parallel(self, bass_line, melody_line):
		"""
		Check if there is parallel motion in octaves or fifths
		in a line of notes
		"""
		
		first_interval = interval.notesToChromatic(bass_line[0], melody_line[0]).semitones % 12
		second_interval = interval.notesToChromatic(bass_line[1], melody_line[1]).semitones % 12
		
		if (first_interval == second_interval) and (second_interval in set([0, 7])):
			return 1
		else:
			return 0
		
		
class ContraryMotion(Constraint):
	"""
	Prefer contrary motion between soprano and bass
	"""
	def not_satisfied(self, MCMC, chain, index_):
		
		bass = MCMC.bass
		
		proposed_note = chain[index_]
		
		# if at first note -> just intervals during and after
		if index_ == 0:
			
			# get the notes needed
			proposed_note_bass = bass[index_]
			next_note_melody = chain[index_ + 1]
			next_note_bass = bass[index_ + 1]
			
			bass_ = [proposed_note_bass, next_note_bass]
			melody_ = [proposed_note, next_note_melody]
			
			# if current and next note is not a rest
			if not isinstance(next_note_bass, (note.Rest)) and \
				(self.not_contrary(bass_, melody_) == 1):
				return 1
			else:
				return 0
			
		else:
			# get the notes needed
			previous_note_melody = chain[index_ - 1]
			previous_note_bass = bass[index_ - 1]
			proposed_note_bass = bass[index_]
			
			bass_ = [previous_note_bass, proposed_note_bass]
			melody_ = [previous_note_melody, proposed_note]

			notes = [previous_note_melody, previous_note_bass, proposed_note_bass, proposed_note]
			
			# if previous note is not a rest
			if not any([isinstance(n, note.Rest) for n in notes]) and \
				(self.not_contrary(bass_, melody_) == 1):
				return 1
			else:
				return 0
		
	
	def profiling(self, MCMC, line):
		
		bass = list(MCMC.bassline.recurse(classFilter=('Note')))
		soprano = list(line.recurse(classFilter=('Note')))
		
		bass_intervals = extract_utils.get_intervals(bass)
		soprano_intervals = extract_utils.get_intervals(soprano)
		
		contrary_count = 0
		for i in np.arange(len(bass_intervals)):
			if np.sign(bass_intervals[i]) == -np.sign(soprano_intervals[i]):
				contrary_count += 1
		return contrary_count / len(soprano_intervals)
	
	
	def not_contrary(self, bass_line, melody_line):
		"""
		Check if there is contrary motion in the bass and melody
		"""
		
		first_dir = np.sign(interval.notesToChromatic(melody_line[0], melody_line[1]).semitones)
		second_dir = np.sign(interval.notesToChromatic(bass_line[0], bass_line[1]).semitones)
		
		if first_dir == -second_dir:
			return 0
		else:
			return 1


class NoteToTonic(Constraint):
	"""
	Each leading note and supertonic should rise/fall
	respectively to the tonic
	"""
	
	def not_satisfied(self, MCMC, chain, index_):
		
		tonic = MCMC.key.getTonic()
		
		# impossible for final note to lead
		if index_ == len(chain) - 1:
			return 0
		
		proposed_note = chain[index_]
		next_note = chain[index_ + 1]
		
		if (index_ <= len(chain) - 1) and \
			(not isinstance(next_note, (note.Rest))) and \
			(self.does_not_lead(proposed_note, next_note, tonic) == 1):
			return 1
		else:
			return 0
			
	def profiling(self, MCMC, line):
		"""
		Profile the marginal progress of this constraint
		"""
		
		tonic = MCMC.key.getTonic()
		
		line = list(line.recurse(classFilter=('Note')))
		line_degrees = [(n.pitch.pitchClass - tonic.pitchClass) % 12 for n in line if not n.isRest]
		
		# initialize to 1 to prevent from dividing by 0
		leading_tones = 1
		leading_tones_resolved = 1
		for i in np.arange(len(line_degrees)-1):
			
			# if a leading note found
			if line_degrees[i] in set([1,2,10,11]) and i <= len(line_degrees) - 1:
				leading_tones += 1
				
				# if that note is resolved
				if line_degrees[i + 1] == tonic.pitchClass:
					leading_tones_resolved += 1
				
		return leading_tones_resolved / leading_tones
			
			
	def does_not_lead(self, note_1, note_2, tonic_pitch):
		"""
		Function to return 1 if `note_1` is a supertonic or
		leading note and `note_2` is NOT a tonic, else 0
		
		Parameters
		----------
		
			note_1 : note.Note
			note_2 : note.Note
			tonic_pitch : pitch.Pitch
		"""
		
		# if note_1 is a supertonic or leading note and note_2 is a tonic
		if ((note_1.pitch.pitchClass - tonic_pitch.pitchClass) % 12 in set([1,2,10,11])) and \
			(note_1.pitch.pitchClass != tonic_pitch.pitchClass):
			return 1             
		else:
			return 0


class LeapThenStep(Constraint):
	"""
	If there is a leap by more than a fourth, follow this by a step
	"""
	
	def not_satisfied(self, MCMC, chain, index_):
		
		
		# impossible for final note to lead
		if index_ >= len(chain) - 2:
			return 0
		
		proposed_note = chain[index_]
		next_note = chain[index_ + 1]
		last_note = chain[index_ + 2]
		
		if (index_ < len(chain) - 2) and \
			(not isinstance(next_note, (note.Rest))) and \
			(not isinstance(last_note, (note.Rest))) and \
			(self.does_not_step_after_leap(proposed_note, next_note, last_note) == 1):
			return 1
		else:
			return 0
			
	def profiling(self, MCMC, line):
		"""
		Profile the marginal progress of this constraint
		"""
		
		line = list(line.recurse(classFilter=('Note')))
		
		# initialize to 1 to prevent division by 0
		jumps = 1
		jumps_not_followed_by_step = 1
		for i in np.arange(len(line) - 2):
			
			if abs(interval.notesToChromatic(line[i], line[i + 1]).semitones) > 4:
				
				# jump
				jumps += 1
				
				# check if jump is followed by a step
				if self.does_not_step_after_leap(line[i], line[i + 1], line[i + 2]) == 1:
					jumps_not_followed_by_step += 1
			
		return 1 - (jumps_not_followed_by_step / jumps)
			
			
	def does_not_step_after_leap(self, note_1, note_2, note_3):
		"""
		Function to return 1 if
			-> the interval between `note_1` and `note_2` is greater than
			a fourth and the interval between `note_2` and `note_3` is not a step
		
		Parameters
		----------
		
			note_1 : note.Note
			note_2 : note.Note
			note_3 : note.Note
		"""
		
		first_interval = abs(interval.notesToChromatic(note_1, note_2).semitones)
		second_interval = abs(interval.notesToChromatic(note_2, note_3).semitones)
		
		
		# if note_1 is a supertonic or leading note and note_2 is a tonic
		if (first_interval > 4) and (second_interval >= 2):
			return 1             
		else:
			return 0


class ReduceRepeated(Constraint):
	"""
	Constraint which encourages movement in the soprano part
	"""
	
	def not_satisfied(self, MCMC, chain, index_):
		
		proposed_note = chain[index_] 
		
		# if at first note -> just interval after
		if index_ == 0:
			
			next_note = chain[index_ + 1]
			
			# if next note is not a rest
			if not isinstance(next_note, (note.Rest)) and \
				(self.is_repeated(proposed_note, next_note) == 1):
				return 1
			else:
				return 0
		
		else:
			
			previous_note = chain[index_ - 1]
			
			if not isinstance(previous_note, (note.Rest)) and \
				(self.is_repeated(previous_note, proposed_note) == 1):
				return 1
			else:
				return 0
			
	def profiling(self, MCMC, line):
		
		line = list(line.recurse(classFilter=('Note')))
		
		profile_array = np.empty((len(line) - 1,))
		for i in np.arange(1, len(line)):
			profile_array[i-1] = self.is_repeated(line[i-1], line[i])
			
		return 1 - np.nansum(profile_array) / (len(line) - 1)
			
	
	def is_repeated(self, note_1, note_2):
		"""
		Function that returns 1 if `note_1` and `note_2` have the same pitch
		
		Parameters
		----------
		
			note_1 : note.Note
			note_2 : note.Note
		"""
		
		if note_1.pitch.name == note_2.pitch.name:
			return 1
		else:
			return 0


class MovementWithinThird(Constraint):
	""" 
	Constraint designed to reduce large jumps
	"""

	def not_satisfied(self, MCMC, chain, index_):
		
		proposed_note = chain[index_]

		# if final index check previous note
		if index_ == len(chain) - 1:
			previous_note = chain[index_ - 1]

			if (not isinstance(previous_note, (note.Rest))) and \
				(self.more_than_third(previous_note, proposed_note) == 1):
				return 1
			else:
				return 0
		
		else:
			next_note = chain[index_ + 1]
			
			if (not isinstance(next_note, (note.Rest))) and \
				(self.more_than_third(proposed_note, next_note) == 1):
				return 1
			else:
				return 0
			
	def profiling(self, MCMC, line):
		"""
		Profile the marginal progress of this constraint
		"""
		
		
		line = list(line.recurse(classFilter=('Note')))
		
		jumps = 0
		for i in np.arange(len(line)-1):
			if abs(interval.notesToChromatic(line[i], line[i + 1]).semitones) > 4:
				jumps += 1
				
		return 1 - (jumps / (len(line)-1))
			
			
	def more_than_third(self, note_1, note_2):
		"""
		Function to return 1 if the interval between `note_1` and
		`note_2` is larger than a major third and 0 otherwise
		
		Parameters
		----------
		
			note_1 : note.Note
			note_2 : note.Note
		"""
		
		# if note_1 is a supertonic or leading note and note_2 is a tonic
		if abs(interval.notesToChromatic(note_1, note_2).semitones) > 4:
			return 1
		else:
			return 0


class CrossConstraint(Constraint):

	def __init__(self, name, constraint_list):
		self.name = name
		self.cl = constraint_list

	def not_satisfied(self, MCMC, chain, index_):
		return np.nanprod([c.not_satisfied(MCMC, chain, index_) for c in self.cl])

	def profiling(self, MCMC, line):
		return np.nanprod([c.profiling(MCMC, line) for c in self.cl])


def create_cross_constraint_dict(constraint_dict, model):
	"""
	Turn a constraint_dict into a larger constraint_dict, including
	cross constraints
	"""
	
	cd_out = {}
	
	for i, k_outer in enumerate(list(constraint_dict.keys())):
		
		cd_out[k_outer] = constraint_dict[k_outer]
		for j, k_inner in enumerate(list(constraint_dict.keys())):
		
			# just lower triangle
			if i < j:
				name_ = k_outer + '/' + k_inner
				
				if model == 'MH':
					cd_out[name_] = CrossConstraint(
						name_, [constraint_dict[k_outer], constraint_dict[k_inner]]
					)
				else:
					cd_out[name_] = gibbs_boolean.CrossConstraint(
						name_, [constraint_dict[k_outer], constraint_dict[k_inner]]
					)
			
	return cd_out  

# trained weights
# NIJ          6.720067
# NIJ/NPI      6.602889
# NIJ/CM       7.351684
# NIJ/NTT      6.749566
# NIJ/LTS      6.011395
# NIJ/RR       7.656894
# NIJ/MWT      6.948376
# NPI          6.044982
# NPI/CM       6.367033
# NPI/NTT      7.165307
# NPI/LTS      6.124189
# NPI/RR       6.930095
# NPI/MWT      5.977234
# CM           3.956699
# CM/NTT       7.115753
# CM/LTS       5.530217
# CM/RR        7.030064
# CM/MWT       6.264896
# NTT          6.689717
# NTT/LTS      6.716246
# NTT/RR       6.208140
# NTT/MWT      6.814695
# LTS          3.973700
# LTS/RR       7.024067
# LTS/MWT      3.802352
# RR           5.215452
# RR/MWT       5.803847
# MWT          3.683069
# MSE        948.166726


		