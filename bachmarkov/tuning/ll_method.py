import bachmarkov
from utils import chord_utils, extract_utils, data_utils
from hmm import hmm
from mh import mh
from tuning import convergence_diagnostics
import importlib
import networkx as nx
import numpy as np
import pandas as pd
import copy
from music21 import *
from scipy.stats.mstats import gmean

import seaborn as sns

import tqdm
from tqdm import trange

class WeightTrainer():
    """
    Given a list of constraints, calculate corresponding weights that minimize the log-likelihood function in the
    true Bach soprano line
    """
    
    def __init__(self, vocal_range, soprano, bass, chords, constraint_dict, T_0, alpha):
        
        self.vocal_range = vocal_range
        self.soprano = list(soprano.recurse(classFilter=('Note','Rest')))
        self.bass = list(bass.recurse(classFilter=('Note', 'Rest')))
        self.bassline = bass
        self.chords = chords
        self.constraint_dict = constraint_dict
        self.T = T_0
        self.alpha = alpha
        
        self.key = bass.analyze('key')
        
    def run(self, n_iter):
        
        scores = np.empty((n_iter, len(self.constraint_dict.keys()) + 1))
        scores_df = pd.DataFrame(scores)
        scores_df.index = np.arange(n_iter)
        scores_df.columns = list(self.constraint_dict.keys()) + ['Log-Lik']
        
        for i in trange(n_iter, desc=' Number of Weightings tried'):
        
            # generate random weight metric
            weight_dict = {k: np.random.randint(1, 1000) for iter_, k in enumerate(list(self.constraint_dict.keys()))}
            scores_df.iloc[i, :-1] = np.array(list(weight_dict.values()))
            
            p_is = np.empty((len(self.bass),))
            
            for j in np.arange(len(self.bass)):
                
                # if rest, rest is only choice
                if self.chords[j] == -1:
                    p_is[j] = 1
                    continue
                
                # possible notes
                possible_notes = chord_utils.random_note_in_chord_and_vocal_range(
                    extract_utils.extract_notes_from_chord(self.chords[j]),
                    self.key,
                    self.vocal_range
                )
                
                # loop through notes in chorale
                note_probabilities_dict = {}
                
                # loop through possible notes
                for k in np.arange(len(possible_notes)):
                    
                    possible_chain = self.soprano[:]
                    possible_chain[j] = possible_notes[k]
        
                    # for each note in chorale calculate p_i
                    constraints_not_met = \
                        [self.constraint_dict[constraint].not_satisfied(self, possible_chain, j) * \
                        weight_dict[constraint] for constraint in list(self.constraint_dict.keys())]
                    
                    note_probabilities_dict[possible_notes[k].nameWithOctave] = np.exp(-np.nansum(constraints_not_met)/self.T)
                    total_sum = np.nansum(list(note_probabilities_dict.values()))
                    note_probabilities_dict = {key : note_probabilities_dict[key] / total_sum for key in list(note_probabilities_dict.keys())}
                

                p_is[j] = note_probabilities_dict.get(self.soprano[j].nameWithOctave, 0.001)
        
            # return scores
            scores_df.loc[i, 'Log-Lik'] = -np.log((np.nanprod(p_is)))
        
        return scores_df
        

class WeightTrainerInner():
    """
    Given a list of constraints, calculate corresponding weights that minimize the log-likelihood function in the
    true Bach inner parts line
    """
    
    def __init__(self, vocal_range_dict, soprano, alto, tenor, bass, chords, constraint_dict, T_0, alpha):
        
        self.vocal_range_dict = vocal_range_dict
        self.soprano = soprano
        self.alto = alto
        self.tenor = tenor
        self.bass = bass
        self.bassline = bass
        self.chords = chords
        self.constraint_dict = constraint_dict
        self.T = T_0
        self.alpha = alpha
        
        self.key = bass.analyze('key')
        
    def run(self, n_iter):
        
        scores = np.empty((n_iter, len(self.constraint_dict.keys()) + 1))
        scores_df = pd.DataFrame(scores)
        scores_df.index = np.arange(n_iter)
        scores_df.columns = list(self.constraint_dict.keys()) + ['Log-Lik']
        
        
        flat_alto = list(self.alto.recurse(classFilter=('Note', 'Rest')))
        flat_tenor = list(self.tenor.recurse(classFilter=('Note', 'Rest')))
        
        for i in trange(n_iter, desc=' Number of Weightings tried'):
            
            # generate random weight metric
            weight_dict = {k: np.random.randint(1, 1000) for iter_, k in enumerate(list(self.constraint_dict.keys()))}
            scores_df.iloc[i, :-1] = np.array(list(weight_dict.values()))
            
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
                        weight_dict[constraint] for constraint in list(self.constraint_dict.keys())]
                    
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
            scores_df.loc[i, 'Log-Lik'] = -np.nansum(np.log(p_is))
        
        return scores_df


def weight_aggregator(chorales, constraint_dict, vocal_range, n_iter):
    """
    Function to train weights over a list of chorales. Each chrorale 
    undergoes a WeightTrainer run for `n_iter` iterations and returns the minimum log likelihood
    """
    
    scores = np.empty((len(chorales), len(constraint_dict.keys()) + 1))
    scores_df = pd.DataFrame(scores)
    scores_df.index = np.arange(len(chorales))
    scores_df.columns = list(constraint_dict.keys()) + ['Log-Lik']
    
    for i, chorale in enumerate(chorales):
        
        chorale_test = WeightTrainer(
            vocal_range,
            extract_utils.to_crotchet_stream(chorale.parts['Soprano']),
            extract_utils.to_crotchet_stream(chorale.parts['Bass']),
            chord_utils.degrees_on_beats(chorale),
            constraint_dict,
            1000,
            0.96
        )

        try:
            chorale_test_out = chorale_test.run(n_iter)
        
            min_log_lik = np.array(chorale_test_out.iloc[chorale_test_out['Log-Lik'].argmax()])
            scores_df.iloc[i, :] = min_log_lik
        except:
            scores_df.iloc[i, :] = np.nan
        
        
    return scores_df


def weight_aggregator_inner(chorales, constraint_dict, vocal_range_dict, n_iter):
    """
    Function to train weights over a list of chorales. Each chrorale 
    undergoes a WeightTrainer run for `n_iter` iterations and returns the minimum log likelihood
    """
    
    scores = np.empty((len(chorales), len(constraint_dict.keys()) + 1))
    scores_df = pd.DataFrame(scores)
    scores_df.index = np.arange(len(chorales))
    scores_df.columns = list(constraint_dict.keys()) + ['Log-Lik']
    
    for i, chorale in enumerate(chorales):
        
        chorale_test = WeightTrainerInner(
            vocal_range_dict,
            extract_utils.to_crotchet_stream(chorale.parts['Soprano']),
            extract_utils.to_crotchet_stream(chorale.parts['Alto']),
            extract_utils.to_crotchet_stream(chorale.parts['Tenor']),
            extract_utils.to_crotchet_stream(chorale.parts['Bass']),
            chord_utils.degrees_on_beats(chorale),
            constraint_dict,
            1000,
            0.96
        )

        try:
            chorale_test_out = chorale_test.run(n_iter)
        
            min_log_lik = np.array(chorale_test_out.iloc[chorale_test_out['Log-Lik'].argmax()])
            scores_df.iloc[i, :] = min_log_lik
        except:
            scores_df.iloc[i, :] = np.nan
        
        
    return scores_df
        

