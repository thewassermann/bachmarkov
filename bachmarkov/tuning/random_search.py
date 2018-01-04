import bachmarkov
from utils import chord_utils, extract_utils, data_utils
from hmm import hmm
from mh import mh
from gibbs import gibbs
import importlib
import networkx as nx
import numpy as np
import pandas as pd
import copy
from music21 import *
from scipy.stats.mstats import gmean
from scipy.stats import norm as normal

import tqdm
from tqdm import trange

from tuning import convergence_diagnostics

import matplotlib.pyplot as plt

class RandomSearch():
    """
    Function to condut a Random Search in order to guage the best weight dictionary
    """
    
    def __init__(self, test_chorales, model, model_type):
        # get dictionary for functional forms
        if model_type == 'MH':
            self.function_dict = model.fitness_function_dict
        else:
            self.function_dict = model.conditional_dict

        self.model_type = model_type

        # ge list of test chorales
        self.test_chorales = test_chorales
        self.weight_dict = self.init_weight_dict()

        if model_type == 'MH':
            self.vocal_range = model.vocal_range
    
    def init_weight_dict(self):
        return {k : 0 for k in list(self.function_dict.keys())}
    
    
    def run(self, n_iter, run_length, walkers):
        
        # set up output dataframe
        out_arr = np.empty((n_iter, len(list(self.function_dict.keys())) + 2))
        out_df = pd.DataFrame(out_arr)
        out_df.index = np.arange(n_iter)
        out_df.columns = list(self.function_dict.keys()) + ['Score', 'PSRF']
        
        # run the search `n_iter` times
        for i in trange(n_iter, desc='Search Iteration', leave=True, position=0):
            
            # set up a new weight dictionary to test
            choices = np.random.choice(100, size=len(list(self.function_dict.keys())))
            weight_list = choices / np.nansum(choices)
            self.weight_dict = {k : weight_list[idx] for idx, k in enumerate(list(self.function_dict.keys()))}
            
            # update output frame for new iteration through chorales
            for key in list(self.weight_dict.keys()):
                out_df.loc[i, key] = self.weight_dict[key]
            
            # set up structure to store score of 
            chorale_stats_PSRF = np.empty((len(self.test_chorales),))
            chorale_stats_score = np.empty((len(self.test_chorales),))
            
            # loop through each chorale
            for j in np.arange(len(self.test_chorales)):
            
                if self.model_type == 'MH':
                    model_ = mh.MH(
                        self.test_chorales[j],
                        self.vocal_range,
                        self.function_dict,
                        weight_dict=self.weight_dict,
                        prop_chords = None,
                    )
                else:
                    model_ = gibbs.GibbsSampler(
                        self.test_chorales[j],
                        extract_utils.to_crotchet_stream(self.test_chorales[j].parts['Soprano']),
                        extract_utils.to_crotchet_stream(self.test_chorales[j].parts['Bass']),
                        chord_utils.degrees_on_beats(self.test_chorales[j]),
                        vocal_range_dict={
                            'Alto' : (pitch.Pitch('g3'), pitch.Pitch('c5')),
                            'Tenor' : (pitch.Pitch('c3'), pitch.Pitch('e4')), 
                        },
                        conditional_dict={
                            'NC' : gibbs.NoCrossing('NC'),
                            'SWM' : gibbs.StepWiseMotion('SWM'),
                            'NPM' : gibbs.NoParallelMotion('NPM'),
                            'OM' : gibbs.OctaveMax('OM')
                        },
                    )
                
                cd = convergence_diagnostics.ConvergenceDiagnostics(
                    model_,
                    self.model_type,
                    run_length,
                    'Tsang Aitken',
                    int(run_length / 2),
                    walkers,
                    tqdm_show=False,
                    plotting=False
                )
                
                # run the profiler for the jth chorale
                chorale_stats_PSRF[j] = cd.PSRF
                chorale_stats_score[j] = model_.run(run_length, True, False).iloc[-1]
                
                
            out_df.loc[i, 'PSRF'] = np.nanmean(chorale_stats_PSRF)
            out_df.loc[i, 'Score'] = np.nanmean(chorale_stats_score)
            
        return out_df