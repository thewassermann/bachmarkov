"""
Utility functions for feature/part extraction
"""

from music21 import *
import numpy as np
import copy


def flattened_to_stream(flat, bassline, out_stream):
	"""
	Function to take in flattened data of just notes and rests
	and transform it into a fully fledged stream
	"""

	# loop through measures
	note_index = 0
	for el in bassline.recurse(skipSelf=True):
		if el == clef.BassClef():
			out_stream.insert(clef.TrebleClef())
		elif isinstance(el, instrument.Instrument):
			out_stream.insert(instrument.Soprano())
		elif isinstance(el, (stream.Measure)):
			# select a random index for a semitone and add to outstream
			m = stream.Measure()
			for measure_el in el:
				if isinstance(measure_el, note.Note):
					out_note = flat[note_index]
					m.insert(measure_el.offset, out_note)
				else:
					m.insert(measure_el.offset, note.Rest(quarterLength=1))
				note_index += 1
			out_stream.insert(el.offset, m)
		elif isinstance (el, (note.Note, note.Rest)):
			continue
		else:
			out_stream.insert(el.offset, copy.deepcopy(el))

	return out_stream


def get_intervals(part_list, start_ix=None, end_ix=None):
	"""
	Function to get intervals between notes in parts

	Parameters
	----------

		part_list : list of music21.note.Note and music21.note.Rest
	"""
    
	no_rests = [x for x in part_list if not x.isRest]
	out_stream = np.empty((len(no_rests),))
	for i, el in enumerate(no_rests):
        
        # first entry has no interval
		if i == 0:
			out_stream[i] = 0
		else:
			prev = no_rests[i-1]
			out_stream[i] = interval.notesToChromatic(prev, el).semitones

	if (start_ix is None) and (end_ix is None):
		return out_stream
	elif (start_ix is not None) and (end_ix is not None):
		return out_stream[start_ix:end_ix]
	else:
		print('Specify both start and end')


def to_crotchet_stream(part, chord_flag=False):
	"""
	Function to take in part of a chorale and turn it into
	a stream of crotchets (quarter notes)

	Parameters
	----------

	    part : music21.stream.Part
	    
	Returns

	    music21.stream.Part exactly the same
	    as the parameter `part` but with only
	    notes on beats
	"""

	# structure for outputting data
	out_stream = stream.Part()

	# recurse through all data in given part
	# and return only notes with whole quarter beats
	for item in part.recurse(skipSelf=True):
            
            # get measure offset
			if isinstance(item, (stream.Measure)):
				measure_offset = item.offset
				m = stream.Measure()

				for measure_item in item:

					if not chord_flag:
                    
						if isinstance(measure_item, (chord.Chord, note.Note)):
							# check offset and beat number
							offset = measure_item.offset
							dur = measure_item.duration.quarterLength

							# on beat note, less than 1 beat
							if offset.is_integer() and dur <= 1.:
								if isinstance(measure_item, (note.Rest)):
									m.insert(offset, note.Rest(quarterLength=1))
								else:
									m.insert(offset, note.Note(measure_item.pitch, quarterLength=1))
	                            
	                        # on beat note, more than one beat
							elif offset.is_integer() and dur > 1.:
								for cnt in np.arange(np.floor(dur)):
									if isinstance(measure_item, (note.Rest)):
										m.insert(offset + cnt, note.Rest(quarterLength=1))
									else:
										m.insert(offset + cnt, note.Note(measure_item.pitch, quarterLength=1))
	                                
	                        # on non-beat note, not covering beat -> skip
	                        # on non-beat note, covering beat
							elif dur >= 1.:
								for cnt in np.arange(np.floor(dur)):
									if isinstance(measure_item, (note.Rest)):
										m.insert(np.ceil(offset) + cnt, note.Rest(quarterLength=1))
									else:
										m.insert(np.ceil(offset) + cnt, note.Note(measure_item.pitch, quarterLength=1))

					else:

						if isinstance(measure_item, (chord.Chord, note.Note)):
							# check offset and beat number
							offset = measure_item.offset
							dur = measure_item.duration.quarterLength

							# on beat note, less than 1 beat
							if offset.is_integer() and dur <= 1.:
								if isinstance(measure_item, (note.Rest)):
									m.insert(offset, note.Rest(quarterLength=1))
								else:
									m.insert(offset, chord.Chord(measure_item.pitches, quarterLength=1))
	                            
	                        # on beat note, more than one beat
							elif offset.is_integer() and dur > 1.:
								for cnt in np.arange(np.floor(dur)):
									if isinstance(measure_item, (note.Rest)):
										m.insert(offset + cnt, note.Rest(quarterLength=1))
									else:
										m.insert(offset + cnt, chord.Chord(measure_item.pitches, quarterLength=1))
	                                
	                        # on non-beat note, not covering beat -> skip
	                        # on non-beat note, covering beat
							elif dur >= 1.:
								for cnt in np.arange(np.floor(dur)):
									if isinstance(measure_item, (note.Rest)):
										m.insert(np.ceil(offset) + cnt, note.Rest(quarterLength=1))
									else:
										m.insert(np.ceil(offset) + cnt, chord.Chord(measure_item.pitches, quarterLength=1))
                                
				# add measure to score
				out_stream.insert(measure_offset, m)
            
            # notes and rests dealt with in measure code
			elif isinstance(item, (chord.Chord, note.Note, note.Rest)):
				continue
			else:
				out_stream.insert(item.offset, item)
                    
	return out_stream

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