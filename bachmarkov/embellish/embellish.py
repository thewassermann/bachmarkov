from  music21 import *
import numpy as np
import pandas as pd
from scipy.stats import chi2

from tqdm import trange
import matplotlib.pyplot as plt

import importlib
import copy

from random import shuffle

from utils import chord_utils, extract_utils, data_utils

import copy

class Embellisher():
	"""
	Class to provide embellishments to individual lines of
	the a chorale. This will convert the parts on beats
	into fully-fledged lines
	"""

	def __init__(self, parts_on_beats, rules_dict, bassline, vocal_range, fermata_layer):

		self.parts_on_beats = parts_on_beats
		self.rules_dict = rules_dict
		self.fermata_layer = fermata_layer
		self.key = parts_on_beats.analyze('key')
		self.bass = bassline
		self.vocal_range = vocal_range
		self.run_part('Soprano')
		self.run_part('Alto')
		self.run_part('Tenor')

	def return_chorale(self):
		"""
		Return the chorale in its current form as a score
		"""
		# conjoin two parts and shows
		s = stream.Score()

		# add in expression
		soprano = stream.Part(self.soprano)
		soprano.id = 'Soprano'

		alto = stream.Part(self.alto)
		alto.id = 'Alto'
		tenor = stream.Part(self.tenor)
		tenor.id = 'Tenor'
		bass = stream.Part(self.bass)
		bass.id = 'Bass'

		s.insert(0, copy.deepcopy(soprano))
		s.insert(0, copy.deepcopy(alto))
		s.insert(0, copy.deepcopy(tenor))
		s.insert(0, copy.deepcopy(bass))
		return s

	def run_part(self, target_part_name):

		target_part = self.parts_on_beats.parts[target_part_name]

		# get order for embellishment functions to run
		embellishment_functions = list(self.rules_dict.keys())

		for embellishment_function in embellishment_functions:

			target_part = self.rules_dict[embellishment_function].embellish(self, target_part, target_part_name)

			# update parts
			if target_part_name == 'Soprano':
				self.soprano = target_part
			elif target_part_name == 'Alto':
				self.alto = target_part
			elif target_part_name == 'Tenor':
				self.tenor = target_part

class Embellishment():

	def __init__(self, name):
		self.name = name

	def embellish(self, embellisher, target_part):
		pass



# class FillInThirds(Embellishment):

# 	def embellish(self, embellisher, target_part, target_part_name):
# 		"""
# 		When there is an interval of a third, fill in with 
# 		two quavers bridging the gap
# 		"""

# 		# just notes and rests
# 		target_part_flat = list(target_part.recurse(classFilter=('Note', 'Rest')))

# 		# # update the fermata layer if needed:
# 		# if len(target_part_flat) != len(embellisher.fermata_layer):
# 		# 	sopr = list(embellisher.soprano.recurse(classFilter=('Note', 'Rest')))
# 		# 	fermata_layer_update = []
# 		# 	for idx in np.arange(len(sopr)):
# 		# 		if sopr[idx].expressions == []:
# 		# 			fermata_layer_update.append(0)
# 		# 		else:
# 		# 			fermata_layer_update.append(1)
# 		# 	embellisher.fermata_layer = fermata_layer_update

# 		# index to keep track of flattened part
# 		note_index = 0

# 		# stream to store output
# 		out_stream = stream.Stream()

# 		for el in target_part.recurse(skipSelf=True):

# 			if isinstance(el, (stream.Measure)):
# 				# select a random index for a semitone and add to outstream
# 				m = stream.Measure()
# 				for measure_el in el:
# 					if isinstance(measure_el, note.Note):
						
# 						# check if third with next note
# 						if note_index < len(target_part_flat) - 1:
# 							next_note = target_part_flat[note_index + 1]

# 							# correct for rests
# 							if (not isinstance(next_note, note.Rest)) and \
# 								(measure_el.duration.quarterLength == 1.) and \
# 								(embellisher.fermata_layer[note_index] != 1):

# 								interval_ = interval.notesToChromatic(measure_el, next_note).semitones
# 								interval_dir = np.sign(interval_)

# 								if (interval_ in set(np.multiply(interval_dir, [3,4]))) and \
# 									(interval_dir != 0):

# 									first_eighth = note.Note(measure_el.pitch, quarterLength=0.5)
# 									generic_second = interval.GenericInterval(np.multiply(interval_dir, 2))
# 									second_eighth_pitch = generic_second.transposePitchKeyAware(measure_el.pitch, embellisher.key)
# 									second_eighth = note.Note(second_eighth_pitch, quarterLength=0.5)
# 									m.insert(measure_el.offset, first_eighth)
# 									m.insert(measure_el.offset + .5, second_eighth)

# 								else:
# 									m.insert(measure_el.offset, measure_el)
# 							else:
# 								m.insert(measure_el.offset, measure_el)
# 						else:
# 							m.insert(measure_el.offset, measure_el)
# 					else:
# 						m.insert(measure_el.offset, note.Rest(quarterLength=1))
# 					note_index += 1
# 				out_stream.insert(el.offset, m)
# 			elif isinstance (el, (note.Note, note.Rest)):
# 				continue
# 			else:
# 				out_stream.insert(el.offset, copy.deepcopy(el))

# 		return out_stream


class FermataFill(Embellishment):

	def embellish(self, embellisher, target_part, target_part_name):
		"""
		Fill in consecutive quarter notes if they both fave fermatas
		"""

		# just notes and rests
		target_part_flat = list(target_part.recurse(classFilter=('Note', 'Rest')))

		# stream to store output
		out_stream = stream.Stream()
		out_flat = []

		# on multiple fermatas do not operate on them sequentially
		skip_flag = False

		for idx in np.arange(len(target_part_flat)):

			# check if the next beat contains a fermata and,
			# if so, carry the fermata through in a single note

			if (embellisher.fermata_layer[idx] == 1) and \
				 not skip_flag and \
				 not target_part_flat[idx].isRest:

				# get the final index of the fermata
				end_idx = self.get_fermata_end_index(idx, embellisher.fermata_layer)

				# quarter note length of help note
				qLength = float(end_idx - idx)
				note_pitch = target_part_flat[idx].pitch
				out_note = note.Note(note_pitch, quarterLength=qLength)
				if target_part_name == 'Soprano':
					out_note.expressions.append(expressions.Fermata())
				out_flat.append(out_note)
				skip_flag = True

			else:
				if (embellisher.fermata_layer[idx] == 0):
					out_flat.append(copy.deepcopy(target_part_flat[idx]))
					skip_flag = False

		return extract_utils.flattened_to_stream(
			out_flat,
			embellisher.parts_on_beats['Bass'],
			out_stream,
			target_part_name
		)


	def get_fermata_end_index(self, start_index, fermata_layer):
		"""
		Function to return the end index of a string of fermatas

		Parameters
		----------

			start_index : int
				index of flattened part where the fermata has started

			fermata_layer : bool list
		"""				
		N = len(fermata_layer) - start_index

		for i in np.arange(N): 

			if not all(np.array(fermata_layer[start_index:(start_index + i)]).astype(bool)):
				return i + start_index - 1 

		return start_index + i + 1


class RandomEmbellishment(Embellishment):
	
	def __init__(self, name, test_chorale, train_chorales):
		self.name = name
		self.test_chorale = test_chorale
		self.train_chorales = train_chorales
		self.embellishment_dict = self.build_embellishment_dict()
		
		
	def build_part_embellishments(self, chorales, part_name):

		interval_record = {}

		for chorale in chorales:

			key = chorale.analyze('key')
			soprano_flat = list(chorale.parts[part_name].recurse(classFilter=('Note', 'Rest')))
			soprano_crotchet = list(extract_utils.to_crotchet_stream(chorale.parts[part_name]).recurse(classFilter=('Note', 'Rest')))
			soprano_flat_rel_pitch = extract_utils.line_to_relative_pitches(chorale.parts[part_name], key)

			# loop through notes
			crotchet_idx = 1

			for i in np.arange(len(soprano_flat)):


				if (crotchet_idx >= len(soprano_crotchet) - 1):
					break
				# get interval between crotchets
				elif not isinstance(soprano_crotchet[crotchet_idx], note.Rest) and \
					not isinstance(soprano_crotchet[crotchet_idx+1], note.Rest): # and \
					note_1 = int(soprano_flat_rel_pitch[crotchet_idx])
					note_2 = int(soprano_flat_rel_pitch[crotchet_idx+1])
				else:
					continue

				if (note_1, note_2) not in interval_record:
					interval_record[(note_1, note_2)] = []

				if soprano_flat[i].quarterLength < 1:

					embellishment_ = []

					# count number of quarters elapsed
					qadd = 0

					for j in np.arange(i, len(soprano_flat)):

						fwd_interval = interval.Interval(soprano_crotchet[crotchet_idx], soprano_flat[j]).semitones


						if qadd + soprano_flat[j].quarterLength > 1:
							ql = 1 - qadd
						else:
							ql = soprano_flat[j].quarterLength

						embellishment_.append((fwd_interval, ql))

						qadd += ql

						if qadd >= 1:
							break

					interval_record[(note_1, note_2)].append(embellishment_)
				else:
					interval_record[(note_1, note_2)].append([0])

				crotchet_idx += 1

		return interval_record
		
		
		
	def build_embellishment_dict(self):
		
		embellishment_dict = {}
		
		for prt in ['Soprano', 'Alto', 'Tenor']:
			embellishment_dict[prt] = self.build_part_embellishments(self.train_chorales, prt)
			
		return embellishment_dict


	def embellish(self, embellisher, target_part, target_part_name):

		interval_emb = self.embellishment_dict[target_part_name]

		sf = list(target_part.recurse(classFilter=('Note', 'Rest')))
		sfrp = extract_utils.line_to_relative_pitches(target_part, target_part.analyze('key'))

		out_list = []

		for i in np.arange(len(sf)-1):
			
			original = sf[i]
			
			# remove any fermatas
			original.expressions = []
			
			if isinstance(sf[i], note.Rest):
				out_list.append(note.Rest(quarterLength=1))
				continue
		#     elif isinstance(sf[i-1], note.Rest):
		#         out_list.append(original)
		#         continue
			elif isinstance(sf[i+1], note.Rest):
				out_list.append(original)
				continue
			elif i == 0:
				out_list.append(original)
				continue
			elif embellisher.fermata_layer[i] == 1:
				out_list.append(original)
				continue
			else:
				
				int1 = (int(sfrp[i]), int(sfrp[i+1]))
				
				# select interval
				if int1 not in interval_emb:
					out_list.append(original)
					continue
					
				prev_intervals = interval_emb[int1]
					
				emb_choice_idx = np.random.randint(len(prev_intervals))
				emb_choice = prev_intervals[emb_choice_idx]
				
				if emb_choice[0] == 0:
					out_list.append(original)
				else:
					for j, n in enumerate(emb_choice):
						trans, length = n
						out_note = original.transpose(trans)
						out_note.quarterLength = length
						
						# set octave
						prev_oct = out_list[-j].octave
						
						# possible_octaves
						low = copy.deepcopy(out_note)
						low.octave = prev_oct - 1
						
						same = copy.deepcopy(out_note)
						same.octave = prev_oct
						
						high = copy.deepcopy(out_note)
						high.octave = prev_oct + 1
						
						possible_notes = [low, same, high]
						
						# filter for range
						possible_notes = [pn for pn in possible_notes \
										  if ((pn.pitch >= embellisher.vocal_range[target_part_name][0]) & (pn.pitch <= embellisher.vocal_range[target_part_name][1]))]
						
						distances = [np.nanmean([abs(interval.Interval(out_list[-1], n).semitones), abs(interval.Interval(n, sf[i+1]).semitones)]) for n in possible_notes]

						out_note = copy.deepcopy(possible_notes[np.argmin(distances)])
						out_list.append(out_note)
					
						
		out_list.append(sf[-1])                
						
		return extract_utils.flattened_to_stream(
			out_list,
			embellisher.parts_on_beats['Bass'],
			stream.Stream(),
			target_part_name,
		)


			



