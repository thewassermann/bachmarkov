"""
Utility functions for feature/part extraction
"""

from music21 import *

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
    
    # create output stream
	outstream = stream.Stream()

	# chordify the chorale
	bass = chorale.parts['Bass']

	# loop through measures
	for msure in bass.getElementsByClass(stream.Measure):
        
        # loop through chords and rests
		for nte in msure.sliceByBeat().getElementsByClass([note.Note, note.Rest]):
            
            # check if on beat
			if nte.offset.is_integer():
				noteout = nte
				noteout.quarterLength = 1.
				outstream.append(noteout)
                
                
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