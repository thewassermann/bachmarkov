import bachmarkov
from utils import chord_utils, extract_utils, data_utils
from hmm import hmm
from mh import mh
import importlib
import networkx as nx
import numpy as np
import pandas as pd
import copy
from music21 import *
from scipy.stats.mstats import gmean
from scipy.stats import norm as normal

import tqdm

import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

from multiprocessing.dummy import Pool as ThreadPool 


class ConvergenceDiagnostics():

	def __init__(self, model, model_type, n_iter, constraint_name, burn_in=0, walkers=1, tqdm_show=True, plotting=True):
		"""
		Class with functionality to present diagnostic information as to the 
		convergence of the MH algorithm, run `n_iter` times

		Parameters
		----------

			model : mh.MH or gibbs.Gibbs
				metropolis hastings or gibbs sampler model to profile

			model_type : string
				`MH` or `Gibbs`

			n_iter : int
				Number of iterations to run the MH algorithm when profiling

			constraint_name : string
				Name of the constraint being tested

			burn_in : int
				Number of initial iterations to discard

			walkers : int
				Number of paths to take
		"""

		self.model = model
		self.model_type = model_type
		self.n_iter = n_iter
		self.constraint_name = constraint_name
		self.burn_in = burn_in
		self.walkers = walkers
		self.tqdm_show = ~tqdm_show
		self.PSRF = self.run_diagnostics(plotting=plotting)

	def run_diagnostics(self, plotting=True):

		# raise error if burn_in greater than/equal to number of iterations
		if self.n_iter <= self.burn_in:
			raise Exception("Number of iterations insufficient for given burn-in")

		# return the output of the MH Algos
		# mh_outputs = np.empty((self.walkers, self.n_iter))

		if plotting:
			fig, axs = plt.subplots(3,1, figsize=(12, 5 * 3))

		#with tqdm.tqdm(total=self.walkers, desc='Walkers Run', disable=self.tqdm_show) as pbar:

		# multithreading
		pool = ThreadPool(self.walkers)
		mh_outputs = pool.map(self.parallel_walk, np.arange(self.walkers))
		pool.close()
		pool.join()
		mh_outputs = np.array(mh_outputs).astype(float)

		if plotting:

			for i, walk in enumerate(np.arange(self.walkers)):
				# running mean plot
				mh_less_burnin = mh_outputs[i, self.burn_in:]
				axs[0].plot(np.arange(self.burn_in, self.n_iter), mh_less_burnin, alpha=0.1, color='blue')
				axs[0].set_xlabel('n Iterations (Minus Burn-In Period)')
				axs[0].set_ylabel('Score Function')
				axs[0].set_title('{} after {} iterations'.format(self.constraint_name, self.n_iter))

				autocorrelation_plot(mh_less_burnin, ax=axs[1], color='blue', alpha=0.25)
				axs[1].set_title('{} ACF'.format(self.constraint_name))

			#pbar.update(1)

		PSRF = self.gelman_rubin(mh_outputs[:, self.burn_in:])

		if plotting:
			# mean point at each iteration 
			axs[0].plot(np.arange(self.burn_in, self.n_iter), np.nanmean(mh_outputs[:, self.burn_in:], axis=0), color='red')
			axs[0].legend(['Gelman-Rubin PSRF : {}'.format(PSRF)])

			# Yu Mykland CUSUM plot
			axs[2] = self.yu_mykland(axs[2], self.n_iter, mh_less_burnin)

			# fig.suptitle(self.model.chorale.metadata.title[:-4].upper(), y=1.02, fontsize=16)
			fig.tight_layout()

		return PSRF


	def parallel_walk(self, dummy):
		"""
		Abstracted in order to run the function in parallel
		"""
		# initialize new starting mh algo for each walker
		if self.model_type == 'MH':
			self.model.init_melody()
		else:
			self.model.alto = self.model.init_part('Alto', self.model.vocal_range_dict['Alto'], self.model.chords)
			self.model.tenor = self.model.init_part('Tenor', self.model.vocal_range_dict['Tenor'], self.model.chords)

		return np.nanmean(self.model.run(self.n_iter, profiling=True, plotting=False).values, axis=1).flatten()



	def gelman_rubin(self, mh_outputs_less_burnin):
		"""
		Function to produce the Gelman Rubin Potential Scale Reduction Factor

		source : http://www.math.pitt.edu/~swigon/Homework/brooks97assessing.pdf
		"""
		n = mh_outputs_less_burnin.shape[1]

		# number of walkers
		m = mh_outputs_less_burnin.shape[0]

		walker_means = np.nanmean(mh_outputs_less_burnin, axis=1)
		walker_second_moments = np.nanmean(mh_outputs_less_burnin**2, axis=1)
		in_walker_variances = np.nanvar(mh_outputs_less_burnin, axis=1)

		B_over_n = np.nanvar(walker_means)
		W = np.nanmean(in_walker_variances)

		V_hat = (((n-1)/n) * W) + B_over_n

		# now find degrees of freedom
		sample_var_V_hat = ((((n - 1)/n)**2) * (1/m) * np.nanvar(in_walker_variances)) + \
			((((m + 1)/(m*n))**2) * (2 /(m-1)) * ((B_over_n * n)**2)) + \
			(
				((2 * (m + 1) * (n - 1)) / (m * (n**2))) * \
				((n/m) * (
							np.cov(in_walker_variances, walker_second_moments)[1,0] - \
							(2 * np.nanmean(walker_means) * np.cov(in_walker_variances, walker_means)[1,0])
						))
			)

		df = (2 * (V_hat**2)) / sample_var_V_hat

		PSRF = np.sqrt((V_hat / W) * (df / (df-2)))
		return round(PSRF, 4)


	def yu_mykland(self, ax, n_iter, mh_outputs_less_burnin):
		"""
		Function to produce the Yu-Mykland CUSUM statistic
		
		source : http://www.math.pitt.edu/~swigon/Homework/brooks97assessing.pdf
		"""
		n = n_iter
		n_0 = len(mh_outputs_less_burnin)

		mu_hat = np.nanmean(mh_outputs_less_burnin)

		s_hats = np.empty((n-n_0, ))
		for i in np.arange(n-n_0):
			s_hats[i] = np.nansum(np.array(mh_outputs_less_burnin[:i])) - (i * mu_hat)

		# plot CUSUM, ideal is smooth
		ax.plot(np.arange(n_0, n), s_hats)
		ax.set_xlabel('n Iterations (Minus Burn-In)')
		ax.set_ylabel('Yu-Mykland S_hat_T')
		ax.set_title('Yu-Mykland CUSUM')

		D_sequence = np.zeros((n - n_0 - 2))
		for j in np.arange(n - n_0 - 2):
			if ((s_hats[j] > s_hats[j+1]) and (s_hats[j+1] < s_hats[j+2])) or \
				((s_hats[j] < s_hats[j+1]) and (s_hats[j+1] > s_hats[j+2])):
				D_sequence[j] = 1

		D_stat = np.nansum(D_sequence) / (n - n_0)
		ax.legend([
			'D-stat : {}; \n 95% CI : {}, {}'.format(
				D_stat,
				round(0.5 - (normal.ppf(0.975) * np.sqrt((1 / (4 * (n - n_0))))), 3),
				round(0.5 + (normal.ppf(0.975) * np.sqrt((1 / (4 * (n - n_0))))), 3),
			)
		])


