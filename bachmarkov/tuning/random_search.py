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

from multiprocessing.dummy import Pool as ThreadPool 

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
            weight_list = np.random.dirichlet(np.ones(len(list(self.function_dict.keys()))))
            self.weight_dict = {k : weight_list[idx] for idx, k in enumerate(list(self.function_dict.keys()))}
            
            # update output frame for new iteration through chorales
            for key in list(self.weight_dict.keys()):
                out_df.loc[i, key] = self.weight_dict[key]
            
            # set up structure to store score of 
            chorale_stats_PSRF = np.empty((len(self.test_chorales),))
            chorale_stats_score = np.empty((len(self.test_chorales),))
            
            # create multithreaded process
            parallel_params = [(tc, run_length, walkers) for tc in self.test_chorales]
            pool = ThreadPool(walkers)
            out_dict = pool.starmap(self.parallel_run, parallel_params)
            pool.close()
            pool.join()
                
            out_df.loc[i, 'PSRF'] = np.nanmean(np.array([float(o['PSRF']) for o in out_dict]))
            out_df.loc[i, 'Score'] = np.nanmean(np.array([float(o['Score']) for o in out_dict]))
            
        return out_df


    def parallel_run(self, test_chorale, run_length, walkers):
        """
        Abstract the running of each chorale so that process can be multithreaded
        """
        # allow certain runs to fail but catch these
        try:
            if self.model_type == 'MH':
                model_ = mh.MH(
                    test_chorale,
                    self.vocal_range,
                    self.function_dict,
                    weight_dict=self.weight_dict,
                    prop_chords = None,
                )
            else:
                model_ = gibbs.GibbsSampler(
                    test_chorale,
                    extract_utils.to_crotchet_stream(test_chorale.parts['Soprano']),
                    extract_utils.to_crotchet_stream(test_chorale.parts['Bass']),
                    chord_utils.degrees_on_beats(test_chorale),
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
            return {
                'PSRF' : cd.PSRF,
                'Score' : model_.run(run_length, True, False).iloc[-1],
            }

        except Exception as e:

            # what happens if a particular run encounters an error
            return {
                'PSRF' : np.nan,
                'Score' : model_.run(run_length, True, False).iloc[-1],
            }



