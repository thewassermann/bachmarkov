"""
Utility functions for feature/part extraction
"""

from music21 import *


def to_crotchet_stream(part_):
	"""
	Function to return a a music21.part on quarter note beats

	Parameters
	----------

		part : music21.part

	Returns
	-------

		outstream : music21.stream
			A stream containing notes sounding on beats
	"""
    
	# create output stream
	outstream = stream.Stream()

	# loop through measures
	for msure in part_.getElementsByClass(stream.Measure):
        
        # loop through chords
		for nte in msure.sliceByBeat().getElementsByClass([note.Note, note.Rest]):
            
            # check if on beat
			if nte.offset.is_integer():
				noteout = nte
				noteout.quarterLength = 1.
				outstream.append(noteout)
                
	return outstream


def extract_bassline(chorale):
	"""
	Function to return a chorale's bassline on each beat, expressed as degrees of the scale

	Parameters
	----------

		chorale : musicxml object

	Returns
	-------

		degree_outstream : list of integers
			Bassline on beats as degrees of a scale
	"""
    # function to extract the bass line on beats from chorale and put into relative ordered pitch notation
    
	outstream = to_crotchet_stream(chorale.parts['Bass'])
                       
	# turn output stream into degrees of the scale
	degree_outstream = []

	# get pitchClass of chorale
	pc = chorale.analyze('key').getTonic().pitchClass

	# convert notes into scale degrees
	for bass in outstream.recurse().getElementsByClass([note.Note, note.Rest]):
		if bass.isRest:
			degree_outstream.append(-1)
		else:
			degree = (bass.pitchClass - pc) % 12
			degree_outstream.append(degree)
    
	return degree_outstream