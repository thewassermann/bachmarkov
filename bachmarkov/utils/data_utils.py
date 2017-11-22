"""
Utility functions to do with reading/writing/splitting data
"""

from music21 import *
import os
import pickle

"""
These functions are for the reading/writing/cleaning of chorales
"""

def partbool(string):
	"""
	Helper function to ensure that the chorales have the
	correct number and type of parts. Returns False if not
	"""
    prts = chorales[string].parts
    if ((prts[0].partName == 'Soprano') &
        (prts[1].partName == 'Alto') &
        (prts[2].partName == 'Tenor') &
        (prts[3].partName == 'Bass')):
        return True
    else:
        return False


def load_clean_chorales():
	"""
	Function to read the clean chorales from memory
	"""
	return chorales = {
		'Major' : read_pickle('Major Chorales'),
		'Minor' : read_pickle('Minor Chorales')
	}

def save_clean_chorales():
	"""
	Function to clean the chorales in the music21 database and save major
	and minor list of musicxml files

	"""
	chor = corpus.getComposer('bach')

	chorales = {}

	# get parsed formats, not just paths [NB: long function (~1min), tqdm used to monitor progress]
	for i in trange(len(chor), desc='# Chorales Parsed'):

	    c = corpus.parse(chor[i])

	    # remove humdrum duplicates
	    if str(c.corpusFilepath)[:-4] != '.krn':
	        chorales[str(c.corpusFilepath)[5:-4]] = c

	fourpart = [c for c in chorales if len(chorales[c].parts) == 4]
	satb = [fp for fp in fourpart if partbool(fp)]

	satb_major = []
	satb_minor = []

	for i in xrange(len(satb)):
	    if i == 47: # use this to remove problematic chorales
	        next 
	    elif chorales[satb[i]].analyze('key').mode == 'major':
	        satb_major.append(chorales[satb[i]])
	    elif chorales[satb[i]].analyze('key').mode == 'minor':
	        satb_minor.append(chorales[satb[i]])

	write_pickle(satb_major, 'Major Chorales All')
	write_pickle(satb_minor, 'Minor Chorales All')

def write_pickle(var, filename):
	"""
	Save file to memory for easy retreival
	"""
    if not os.path.isdir(os.path.expanduser('~') + '/.pickles'):
        # Use mkdir so we don't get a failure if the dir is created by another process
        os.system('mkdir ~/.pickles/')
    with open(os.path.expanduser('~') + '/.pickles/' + filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


def read_pickle(filename):
	"""
	Load file from memory quickly
	"""
    with open(os.path.expanduser('~') + '/.pickles/' + filename, 'rb') as handle:
        return pickle.load(handle)


"""
These functions are for preparing the chorales for modelling processes
"""