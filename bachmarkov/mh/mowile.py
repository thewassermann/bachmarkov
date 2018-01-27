import numpy as np
import pandas as pd
from mh import mh_boolean
from gibbs import gibbs_boolean
from utils import chord_utils, extract_utils, data_utils

from music21 import *
from tqdm import trange
from scipy.spatial import distance
import importlib

class MOWILE():
    """
    Class to implement the MOre WIth LEss algorithm, designed to search
    through a high parameter subspace with few iterations
    
    The algorithm is split into 3 stages:
        - Sampling phase (Exploration)
        - Subspace selection (Exploitation)
        - Restarts
        
    source:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.151.7075&rep=rep1&type=pdf
        
        
    Parameters
    ----------
    
        chorale :  music21.stream.Score
    
        budget : int
            number of search iterations to perform (rec. [50, 200])
            
        n_restarts : int
            number of restarts to perform (rec. [1, 3])
            
        beta: int
            multiplier of number of samples to generate (rec. 10)
            
        alpha : float
            shrinkage parameter for subspaces
            
        weight_upper_bound : int
            upper bound for parameter space
            
        model : mh_boolean.MHBooleanSampler, gibbs_boolean.GibbsBooleanSampler
        
            
    """
    
    def __init__(self, chorale, budget, n_restarts, beta, alpha, weight_upper_bound, model):
        self.chorale = chorale
        self.budget = budget
        self.n_restarts = n_restarts
        self.beta = beta
        self.alpha = alpha
        self.weight_upper_bound = weight_upper_bound
        self.model = model
        
        # initializing parameters
        self.I = np.zeros((1, len(list(self.model.constraint_dict.keys()))))
        
        # derived parameters
        self.k = int(0.075 * self.budget)
        self.n_steps_between_restarts = int(self.budget / (self.k * (self.n_restarts + 1)))
        self.delta = self.alpha **(-(1/self.n_steps_between_restarts))
        
    # function to minimize
    def min_function(self, b, other_weights):
        """
        Helper function to maximize during the exploration phase
        
        Parameters
        ----------
            
            b : np.array (1 x number of constraints)
            other_weights : np.array (_ x number of constraints )
        """
        
        distances = np.empty((other_weights.shape[0],))
        
        # compare b with each other in sample
        for i in np.arange(other_weights.shape[0]):
            
            # calculate euclidian distance between each
            distances[i] = distance.euclidean(b, other_weights[i, :])
            
        return np.nanmin(distances)
            
        
    def exploration(self, lower_left_coords, upper_bounds):
        """
        Exploration phase of the parameter search
        """
        
        # get samples : C
        # each row is a sample
        # each column is the weight for a different constraint
        constraint_list = list(self.model.constraint_dict.keys())
        C = np.random.uniform(size=(int(self.beta * self.k),len(constraint_list)))
        
        # rescale
        C = (C * upper_bounds) + lower_left_coords
        
        # hold set of bs -> weights to test
        bs = []
    
        # add constraints that are the most distant to previous iterations
        for i in np.arange(self.k):
            
            # min comparison dataframe
            sample_mins = np.empty(C.shape[0])
            
            # loop through C and compare to previous instances
            for j in np.arange(C.shape[0]):
                sample_mins[j] = self.min_function(C[j, :], self.I)
                
            # add the largest to I
            max_arg = np.argmax(sample_mins)
            self.I = np.vstack((self.I, C[max_arg, :]))
            bs.append(C[max_arg, :])
            
            # drop from C
            C = np.delete(C, max_arg, 0)
            
        return np.vstack(bs)

    
    def exploitation(self):
        """
        Explotation phase of parameter search
        """

        coords_ = np.zeros((1, len(list(self.model.constraint_dict.keys()))))
        dim_lens = np.ones((1, len(list(self.model.constraint_dict.keys())))) * self.weight_upper_bound
        
        # explore smaller and smaller subspaces until
        # a restart is needed
        for i in np.arange(self.n_steps_between_restarts):
            
            possible_points = self.exploration(coords_, dim_lens)
            points_score = np.empty((possible_points.shape[0],))
            
            for j in np.arange(possible_points.shape[0]):
            
                # find the likelihood under the model
                model_ = self.model
                model_.weight_dict = {k: possible_points[j, idx] for idx, k in enumerate(list(self.model.constraint_dict.keys()))}
                points_score[j] = model_.log_likelihood()
            
            # get new lower left coords and 
            center_coords = possible_points[np.argmin(points_score), :]
            dim_lens = np.array([dl * (self.alpha ** (1/ len(list(self.model.constraint_dict.keys())))) for dl in dim_lens])
            coords_ = center_coords - (dim_lens/2)
            
            # avoid negative coordinates
            coords_[coords_ < 0] = 0
            coords_[coords_ >= self.weight_upper_bound] = self.weight_upper_bound
            
        # return optimal weights for explotation run
        # tuple of (weights, -loglikelihood)
        best_weights = possible_points[np.argmin(points_score), :]
        best_weights[best_weights < 0] = 0
        best_weights[best_weights >= self.weight_upper_bound] = self.weight_upper_bound
        return (best_weights, np.nanmin(points_score))
    
    def run(self):
        """
        
        Run the full algorithm
        """
        
        out_array = np.empty((self.n_restarts, len(list(self.model.constraint_dict.keys())) + 1))
        out_df = pd.DataFrame(out_array)
        out_df.columns = list(self.model.constraint_dict.keys()) + ['Log-Lik']
        
        # iterations of restarts
        for i in trange(self.n_restarts, desc='Restart Count'):
            weights, score = self.exploitation()
            out_df.iloc[i, :-1] = weights
            out_df.loc[i, 'Log-Lik']
            
        return out_df.iloc[np.argmin(out_df['Log-Lik']), :-1]



def MOWILECV(chorales, kfolds, budget, n_restarts, beta, alpha, weight_upper_bound, cd, mh_iter):
    """
    Perform cross validation to see which weights work best out of sample
    """
    # divide chorales into `k` groups
    chorale_index_groups = np.array_split(np.arange(len(chorales)), kfolds)

    fold_weights_and_MSE = np.empty((kfolds, len(list(cd.values())) + 1))
    
    # loop though folds
    for i in trange(kfolds, desc='Folds elapsed'):
        
        # split into testing and training sets
        testing_ = [chorales[chorale_index] for chorale_index in chorale_index_groups[i]]
        training_indexes = set(np.arange(len(chorales))) - set(chorale_index_groups[i])
        training_ = [chorales[chorale_index] for chorale_index in training_indexes]
        
        # structure to store weights from runs
        fold_training = np.empty((len(training_), len(list(cd.values()))))
        fold_testing = np.empty((len(testing_), 1))
        
        # for each training chorale perform MOWLIE weight search
        for j in np.arange(len(training_)):
            train_mh = mh_boolean.MCMCBooleanSampler(
                training_[j].parts['Bass'],
                (pitch.Pitch('c4'), pitch.Pitch('g5')),
                chord_utils.degrees_on_beats(training_[j]),
                cd_mh,
                extract_utils.extract_fermata_layer(
                    extract_utils.to_crotchet_stream(training_[j].parts['Soprano'])
                ),
                100, # Starting Temperature
                0.925045, # Cooling Schedule
                0.7, # prob of Local Search
                progress_bar_off=True,
                thinning=1000
            )
            train_mowlie = MOWILE(training_[j], budget, n_restarts, beta, alpha, weight_upper_bound, train_mh)
            out_mowlie = np.array(train_mowlie.run())
            fold_training[j, :] = out_mowlie
            
        test_weights = np.nanmean(fold_training, axis=0)
        
        # aggregate weights over chorales
        for k in np.arange(len(testing_)):
            
            test_mh = mh_boolean.MCMCBooleanSampler(
                testing_[k].parts['Bass'],
                (pitch.Pitch('c4'), pitch.Pitch('g5')),
                chord_utils.degrees_on_beats(testing_[k]),
                cd_mh,
                extract_utils.extract_fermata_layer(
                    extract_utils.to_crotchet_stream(testing_[k].parts['Soprano'])
                ),
                100, # Starting Temperature
                0.925045, # Cooling Schedule
                0.7, # prob of Local Search
                weight_dict={dict_key : test_weights[idx] for idx, dict_key in enumerate(list(cd.keys()))},
                progress_bar_off=True,
                thinning=1000
            )
            
            test_mh.run(mh_iter, False, False)
            
            derived_ll = test_mh.log_likelihood()
            
            # true bach
            test_mh.soprano = extract_utils.to_crotchet_stream(testing_[k].parts['Soprano'])
            true_ll = test_mh.log_likelihood()
            
            fold_testing[k] = (derived_ll - true_ll)**2
            
        fold_weights_and_MSE[i, -1] = np.nansum(fold_testing)/len(testing_)
        fold_weights_and_MSE[i, :-1] = test_weights
        
        
    # fold weights and mse to df
    out_df = pd.DataFrame(fold_weights_and_MSE)
    out_df.columns = list(cd.keys()) + ['MSE']
    out_df.index = np.arange(kfolds)
    
    return out_df