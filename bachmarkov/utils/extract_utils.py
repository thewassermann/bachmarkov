"""
Utility functions for feature/part extraction
"""

from music21 import *
import numpy as np
import copy

from utils import chord_utils, extract_utils, data_utils


# def flattened_to_stream(flat, bassline, out_stream, part_, fermata_layer=None):
# 	"""
# 	Function to take in flattened data of just notes and rests
# 	and transform it into a fully fledged stream
# 	"""

# 	# loop through measures
# 	note_index = 0

# 	# offset for held notes
# 	held_offset = 0
# 	for el in bassline.recurse(skipSelf=True):
# 		if el == clef.BassClef():
# 			if (part_ == 'Soprano') or (part_ == 'Alto'):
# 				out_stream.insert(clef.TrebleClef())
# 			else:
# 				out_stream.insert(clef.Treble8vbClef())
# 		elif isinstance(el, instrument.Instrument):
# 			if part_ == 'Soprano':
# 				out_stream.insert(instrument.Soprano())
# 			elif part_ == 'Alto':
# 				out_stream.insert(instrument.Alto())
# 			elif part_ == 'Tenor':
# 				out_stream.insert(instrument.Tenor())
# 		elif isinstance(el, (stream.Measure)):
# 			# select a random index for a semitone and add to outstream
# 			m = stream.Measure()
# 			for measure_el in el:

# 				if held_offset == 0:

# 					if isinstance(measure_el, note.Note):
# 						in_note = flat[note_index]
# 						if (fermata_layer is not None):
# 							if (fermata_layer[note_index] == 1) and (in_note.expressions == []):
# 								in_note.expressions.append(expressions.Fermata())
# 						m.insert(measure_el.offset, in_note)

# 					else:
# 						# in_note = note.Rest(quarterLength=1.)
# 						in_note = note.Rest(quarterLength=flat[note_index].duration.quarterLength)
# 						if (fermata_layer is not None):
# 							if (fermata_layer[note_index] == 1) and (in_note.expressions == []):
# 								in_note.expressions.append(expressions.Fermata())
# 						m.insert(measure_el.offset, in_note)

# 					# number of beats to be held 
# 					held_offset = flat[note_index].duration.quarterLength - 1
# 					note_index += 1

# 				else:	
# 					held_offset -= 1

# 			out_stream.insert(el.offset, m)
# 		elif isinstance (el, (note.Note, note.Rest)):
# 			continue
# 		else:
# 			out_stream.insert(el.offset, copy.deepcopy(el))

# 	return out_stream


def flattened_to_stream(flat, bassline, out_stream, part_, fermata_layer=None):
	"""
	Function to take in flattened data of just notes and rests
	and transform it into a fully fledged stream
	"""

	# loop through measures
	note_index = 0

	# offset for held notes
	held_offset = 0
	for el in bassline.recurse(skipSelf=True):
		if el == clef.BassClef():
			if (part_ == 'Soprano') or (part_ == 'Alto'):
				out_stream.insert(clef.TrebleClef())
			else:
				out_stream.insert(clef.Treble8vbClef())
		elif isinstance(el, instrument.Instrument):
			if part_ == 'Soprano':
				out_stream.insert(instrument.Soprano())
			elif part_ == 'Alto':
				out_stream.insert(instrument.Alto())
			elif part_ == 'Tenor':
				out_stream.insert(instrument.Tenor())
		elif isinstance(el, (stream.Measure)):
			# select a random index for a semitone and add to outstream
			m = stream.Measure()

			# get length of measure
			measure_length = el.duration.quarterLength

			length_counter = 0

			for note_ in flat[note_index:]:

				if fermata_layer is not None:
					if (fermata_layer[note_index] == 1) and (note_.expressions == []):
						note_.expressions.append(expressions.Fermata())

				# check if adding would exceed measure_length
				if note_.duration.quarterLength + length_counter <= measure_length:
					m.insert(length_counter, copy.deepcopy(note_))
					length_counter += flat[note_index].duration.quarterLength
					note_index += 1
				else:
					break

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
		# if list is empty
		if len(out_stream[start_ix:end_ix]) == 0:
			return np.zeros((end_ix - start_ix))
		else:
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
					
						if isinstance(measure_item, (chord.Chord, note.Note, note.Rest)):
							# check offset and beat number
							offset = measure_item.offset
							dur = measure_item.duration.quarterLength

							# on beat note, less than 1 beat
							if offset.is_integer() and dur <= 1.:
								if isinstance(measure_item, (note.Rest)):
									in_note = note.Rest(quarterLength=1)
									in_note.expressions = measure_item.expressions
									m.insert(offset, in_note)
								else:
									in_note = note.Note(measure_item.pitch,quarterLength=1)
									in_note.expressions = measure_item.expressions
									m.insert(offset, in_note)
								
							# on beat note, more than one beat
							elif offset.is_integer() and dur > 1.:
								for cnt in np.arange(np.ceil(dur)):
									if isinstance(measure_item, (note.Rest)):
										in_note = note.Rest(quarterLength=1)
										in_note.expressions = measure_item.expressions
										m.insert(offset + cnt, in_note)
									else:
										in_note = note.Note(measure_item.pitch, quarterLength=1)
										in_note.expressions = measure_item.expressions
										m.insert(offset + cnt, in_note)
									
							# on non-beat note, not covering beat -> skip
							# on non-beat note, covering beat
							elif dur >= 1.:
								for cnt in np.arange(np.ceil(dur)):
									if isinstance(measure_item, (note.Rest)):
										in_note = note.Rest(quarterLength=1)
										in_note.expressions = measure_item.expressions
										m.insert(np.ceil(offset) + cnt, in_note)
									else:
										in_note = note.Note(measure_item.pitch, quarterLength=1)
										in_note.expressions = measure_item.expressions
										m.insert(np.ceil(offset) + cnt, in_note)

					else:

						if isinstance(measure_item, (chord.Chord, note.Note, note.Rest)):
							# check offset and beat number
							offset = measure_item.offset
							dur = measure_item.duration.quarterLength

							# on beat note, less than 1 beat
							if offset.is_integer() and dur <= 1.:
								if isinstance(measure_item, (note.Rest)):
									in_chord = note.Rest(quarterLength=1)
									in_chord.expressions = measure_item.expressions
									m.insert(offset, in_chord)
								else:
									in_chord = chord.Chord(measure_item.pitches, quarterLength=1)
									in_chord.expressions = measure_item.expressions
									m.insert(offset, in_chord)
								
							# on beat note, more than one beat
							elif offset.is_integer() and dur > 1.:
								for cnt in np.arange(np.ceil(dur)):
									if isinstance(measure_item, (note.Rest)):
										in_chord = note.Rest(quarterLength=1)
										in_chord.expressions = measure_item.expressions
										m.insert(offset + cnt, in_chord)
									else:
										in_chord = chord.Chord(measure_item.pitches, quarterLength=1)
										in_chord.expressions = measure_item.expressions
										m.insert(offset + cnt, in_chord)
									
							# on non-beat note, not covering beat -> skip
							# on non-beat note, covering beat
							elif dur >= 1.:
								for cnt in np.arange(np.ceil(dur)):
									if isinstance(measure_item, (note.Rest)):
										in_chord = note.Rest(quarterLength=1)
										in_chord.expressions = measure_item.expressions
										m.insert(np.ceil(offset) + cnt, in_chord)
									else:
										in_chord = chord.Chord(measure_item.pitches, quarterLength=1)
										in_chord.expressions = measure_item.expressions
										m.insert(np.ceil(offset) + cnt, in_chord)
								
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
			degree = (bass.pitch.pitchClass - pc) % 12
			degree_outstream.append(degree)
	
	return degree_outstream


def extract_fermata_layer(part_):
	"""
	Take in a music21.stream.part and get the fermatas on particular notes
	"""
	fermata_layer = []
	for note_ in part_.recurse(classFilter=('Note', 'Rest')):
		if note_.expressions == []:
			fermata_layer.append(0)
		else:
			fermata_layer.append(1)

	return fermata_layer


def extract_notes_from_chord(chord_str):
	"""
	Function to retreive list of notes from chord string

	e.g. `0/4/7` -> [0,4,7]

	Parameters
	----------

		chord_str : string or int (-1 for rests)
	"""

	# if rest
	if chord_str == -1:
		return -1
	else:
		return [int(x) for x in chord_str.split('/')]


def rel_pitch_melody_to_line(melody, bass):
	
	vocal_range = (pitch.Pitch('c4'), pitch.Pitch('g5'))
	
	out_flat = []
	
	# loop through melody to produce output stream
	for i in np.arange(len(melody)):
		
		# if a rest
		if isinstance(melody[i], note.Rest):
			out_flat.append(note.Note, quarterLength=1)
			continue
		
		# if first note choose random octave within range for note
		if i == 0:
			possible_starting_notes = chord_utils.random_note_in_chord_and_vocal_range(
				[float(melody[i])],
				bass.analyze('key'),
				vocal_range,
			)
			out_flat.append(np.random.choice(possible_starting_notes))
		else:
			possible_next_notes = chord_utils.random_note_in_chord_and_vocal_range(
				[float(melody[i])],
				bass.analyze('key'),
				vocal_range,
			)
			
			# if two otpions, choose the one closer to the middle of the vocal range
			if len(possible_next_notes) > 1:
				
				dist_ = [abs(interval.notesToChromatic(pnn, out_flat[i-1]).semitones) for pnn in possible_next_notes]
				out_flat.append(possible_next_notes[np.argmin(dist_)])
			else:
				out_flat.append(possible_next_notes[0])
			
	return out_flat


def line_to_relative_pitches(line, key):
	"""
	Convert a music21.part into a list of notes expressed as
	relative pitches
	"""
	
	line = extract_utils.to_crotchet_stream(line)
	
	line = list(line.recurse(classFilter=('Note', 'Rest')))
	
	outline = []
	
	# get tonic key as a `pitchClass`
	tonic = key.getTonic().pitchClass
	
	for note_ in line:
		
		if isinstance(note_, note.Rest):
			outline.append(str(-1))
		else:
			outline.append(str((note_.pitch.pitchClass - tonic) % 12))
			
	return outline


def embellisher_to_midi(out_name, test_emb):
	mf = midi.MidiFile()
	mf.ticksPerQuarterNote = 1024 # cannot use: 10080
	mf.tracks.append(test_emb.soprano.write('midi'))
	mf.tracks.append(test_emb.alto.write('midi'))
	mf.tracks.append(test_emb.tenor.write('midi'))
	mf.tracks.append(test_emb.bass.write('midi'))

	melody = test_emb.soprano
	alto = test_emb.alto
	tenor = test_emb.tenor
	bass = test_emb.bass

	# conjoin two parts and shows
	s = stream.Score()

	out_stream = stream.Stream()
	soprano_flat = list(melody.recurse(classFilter=('Note', 'Rest')))
	melody = extract_utils.flattened_to_stream(
		soprano_flat,
		test_emb.bass,
		out_stream,
		'Soprano',
		test_emb.fermata_layer
	)

	s.insert(0, stream.Part(melody))
	s.insert(0, stream.Part(alto))
	s.insert(0, stream.Part(tenor))
	s.insert(0, stream.Part(bass))
	s.write('midi', '{}.midi'.format(out_name))
	