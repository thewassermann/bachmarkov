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

class Embellisher():
	"""
	Class to provide embellishments to individual lines of
	the a chorale. This will convert the parts on beats
	into fully-fledged lines
	"""

	def __init__(self, target_part_name, parts_on_beats, rules_dict):

		self.target_part_name = target_part_name
		self.parts_on_beats = parts_on_beats
		self.target_part = self.parts_on_beats.parts[target_part_name]
		self.rules_dict = rules_dict
		self.key = parts_on_beats.analyze('key')

	def run(self):

		# get order for embellishment functions to run
		embellishment_functions = list(self.rules_dict.keys())
		# NB inplace
		shuffle(embellishment_functions)

		for embellishment_function in embellishment_functions:
			self.target_part = self.rules_dict[embellishment_function].embellish(self)

		return self.target_part


class Embellishment():

	def __init__(self, name):
		self.name = name

	def embellish(self, embellisher):
		pass



class FillInThirds(Embellishment):

	def embellish(self, embellisher):
		"""
		When there is an interval of a third, fill in with 
		two quavers bridging the gap
		"""

		# just notes and rests
		target_part_flat = list(embellisher.target_part.recurse(classFilter=('Note', 'Rest')))

		# index to keep track of flattened part
		note_index = 0

		# stream to store output
		out_stream = stream.Stream()

		for el in embellisher.target_part.recurse(skipSelf=True):

			if isinstance(el, (stream.Measure)):
				# select a random index for a semitone and add to outstream
				m = stream.Measure()
				for measure_el in el:
					if isinstance(measure_el, note.Note):
						
						# check if third with next note
						if note_index < len(target_part_flat) - 1:
							next_note = target_part_flat[note_index + 1]
							if interval.notesToChromatic(measure_el, next_note).semitones in set([3,4]):
								first_eighth = note.Note(measure_el.pitch, quarterLength=0.5)
								generic_second = interval.GenericInterval(2)
								second_eighth_pitch = generic_second.transposePitchKeyAware(measure_el.pitch, embellisher.key)
								second_eighth = note.Note(second_eighth_pitch, quarterLength=0.5)
								m.insert(measure_el.offset, first_eighth)
								m.insert(measure_el.offset + .5, second_eighth)
							else:
								m.insert(measure_el.offset, measure_el)
						else:
							m.insert(measure_el.offset, measure_el)
					else:
						m.insert(measure_el.offset, note.Rest(quarterLength=1))
					note_index += 1
				out_stream.insert(el.offset, m)
			elif isinstance (el, (note.Note, note.Rest)):
				continue
			else:
				out_stream.insert(el.offset, copy.deepcopy(el))

		return out_stream			
