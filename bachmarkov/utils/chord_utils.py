"""
Utility functions for dealing with chords
"""

from music21 import *

import numpy as np
from utils import extract_utils
import copy


def relOrderedPitchClassesString(orderedPitches, key):
	"""
	Present chords as collection of scale degrees in unique way
	for easier parsing 

	Parameters
	----------

		orderedPitches : music21.chord.orderedPitchClasses
			List of ordered pitch classes for chords in chorales

		key : music21.keySignature
			key signature of chorale
	

	Returns
	-------

		relativeDegrees : list of integers representing
			degrees of the scale represented in a chord

	"""
    
	# get tonic key as a `pitchClass`
	tonic = key.getTonic().pitchClass

	# get relative ordered pitches
	relOrderedPitches = sorted([(pitch - tonic) % 12 for pitch in orderedPitches])

	# convert to splittable string
	relOrderedPitchesStrings = '/'.join(str(p) for p in relOrderedPitches)
	return(relOrderedPitchesStrings)


def degrees_on_beats(chorale):
	"""
	Function to take in chorales and return the relative degrees
	of the scale for each chord on beats

	Parameters
	----------

		chorale : musicxml object
			A single Bach chorale

	Returns
	-------

		list of strings of unique chordal indentifiers
	"""
    
	# key signature
	key = chorale.analyze('key') # may need to put this elsewhere to deal with modulation

	# create output stream
	outstream = stream.Stream()

	# chordify the chorale
	cc = chorale.chordify() # <- will be replaced by predicted chords

	outstream = extract_utils.to_crotchet_stream(cc, chord_flag=True)
    
	# roman numeral output stream
	rn_outstream = []

	# convert chords to roman numerals
	for bc in outstream.recurse().getElementsByClass([chord.Chord, note.Rest]):
		if bc.isRest:
		    bc = -1
		else:
		    bc = relOrderedPitchClassesString(bc.orderedPitchClasses, key)
		rn_outstream.append(bc)
    
	return rn_outstream


def random_note_in_chord_and_vocal_range(relPitchList, key, vocal_range):
	"""
	Function to return a random note in the chord within a given range
	"""

	# for rests, can only be a rest
	if relPitchList == -1:
		return [note.Rest()]

	# extract_range
	lowest_note, highest_note = vocal_range

	# turn relative Pitch into a note.Note
	notes = []


	for relPitch in relPitchList:
		if relPitch != 0:
			interval_ = interval.ChromaticInterval(relPitch)
			notes.append(note.Note(interval_.transposePitch(key.getTonic()), quarterLength=1))
		else:
			# needed as no transposition by 0 possible
			notes.append(note.Note(key.getTonic(), quarterLength=1))
	        
	# get all notes in right octave
	octave_corrected_notes = []
	for n in notes:

		# possible octaves
		low_octave = lowest_note.octave
		high_octave = highest_note.octave
        
		# select random octave
		# if only one choice available
		octave_choice = np.arange(low_octave, high_octave + 1)

		# loop throughpossible octaves
		for o in octave_choice:
			note_ = copy.deepcopy(n)
			note_.octave = o
			if (note_.pitch >= lowest_note) and (note_.pitch <= highest_note):
				octave_corrected_notes.append(note_)

	return octave_corrected_notes


# def recursed_to_accidentals():
# 	"""
# 	takes a recursed part and returns better accidental choices
# 	"""

