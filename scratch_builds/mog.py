import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
from tqdm import tqdm
from sklearn.datasets import load_iris
from matplotlib.patches import Ellipse
import warnings
from scipy.stats import multivariate_normal


def argmax(list):
    """
    Quick and dirty argmax implementation.
    """
    index = 0
    max_c = list[0]
    for i in range(1, len(list)):
        if list[i] > max_c:
            index = i
            max_c = list[i]
    return index


def mv_gauss(x, mean_vec, cov_matrix):
    """
    Semi-safe multivariate Gaussian pdf calculator.
    """
    n = len(x)
    det = np.linalg.det(cov_matrix)
    noise = 0.01
    if det == 0:  # Not invertible
        while det == 0 and noise <= 1:
            id_noise = np.identity(len(x))*noise
            det = np.linalg.det(cov_matrix + id_noise)
            inv = np.linalg.inv(cov_matrix + id_noise)
            noise += 0.05
    else:
        inv = np.linalg.inv(cov_matrix)
    return 1/((2*math.pi)**(n/2) * det**0.5) * np.exp(-0.5 * np.dot(np.dot((x-mean_vec).T, inv), (x-mean_vec)))


class Gauss(object):
    """
    An object parameterizing a gaussian.
    """

    def __init__(self, y, cov):
        self.y = y
        self.cov = cov

    def __str__(self):
        return f"Mean(s) : {self.y} COV: {self.cov}"


class MOG(object):
    """
    Estimates a GMM model using the EM algorithm.
    """

    def __init__(self, k, dim, progress=True):
        self.weights = []  # list with k entries
        self.gaussians = []  # list with k entries
        self.k = k
        self.dim = dim
        self.progress = progress

    def expectation(self, X):
        """
        Calculates expectation step for each datapoint in X.

        Parameters:
        X       -- A np.array. Each array in X needs to be of shape (-1,1)
        """
        prob_matrix = []
        log_l = []
        member_expectation = []
        for n in range(len(X)):
            x = X[n]
            numerators = []

            for gaussian, weight in zip(self.gaussians, self.weights):
                cond_prob = mv_gauss(x, gaussian.y, gaussian.cov) * weight
                numerators.append(cond_prob)

            PX = sum(numerators)
            cond_probs = [n/PX for n in numerators]
            prob_matrix.append(cond_probs)
            log_l.append(np.log(sum(numerators))[0][0])
            member_expectation.append(argmax(cond_probs))
        return prob_matrix, sum(log_l), member_expectation

    def maximization(self, X, prob_matrix):
        """
        Calculates a maximization step for all gaussians in self.gaussians.

        Parameters:
        X               -- A np.array. Each array in X needs to be of shape (-1,1)
        prob_matrix     -- a list. Each entry in prob_matrix is a n(Gaussians) dimensional list
                           containing for each entry in X the n(Gaussian) relative likelihood of
                           belonging to that Gaussian.
        """
        for index, gaussian in enumerate(self.gaussians):
            cond_probs = [p[index] for p in prob_matrix]

            denominator = sum(cond_probs)
            update_mean = sum([p*x for p, x in zip(cond_probs, X)])/denominator

            update_cov = np.abs(np.sum([p*cov for p, cov in zip(
                cond_probs, [(x-gaussian.y).dot((x-gaussian.y).T) for x in X])], 0)/denominator)
            update_p = denominator/len(X)

            gaussian.y = update_mean
            gaussian.cov = update_cov
            self.weights[index] = update_p

    def fit(self, X, iter):
        """
        Creates k gaussian objects and repeats the
        expectation and maximization steps iter times.

        Parameters:
        X       -- A np.array. Each array in X needs to be of shape (-1,1)
        iter    -- number of iterations to perform. 
        """
        for _ in range(self.k):
            rand_pick = random.choice(X)
            self.gaussians.append(Gauss(rand_pick, rand_pick.dot(
                rand_pick.T) + (np.identity(self.dim)*0.05)))
            self.weights.append(1/self.k)
            # print(self.gaussians[-1])

        expectations = []
        means = []
        covs = []
        iterator = tqdm(range(iter)) if self.progress else range(iter)
        for i in iterator:
            means.append([g.y for g in self.gaussians])
            covs.append([g.cov for g in self.gaussians])
            prob_matrix, log_likelihood, member_exp = self.expectation(X)
            if self.progress:
                print(f"Iteration: {i} Log-Likelihood: {log_likelihood}")
            self.maximization(X, prob_matrix)
            expectations.append(member_exp[:])
        return expectations, means, covs

    def calculate_loss(self, X):
        """
        Returns -1 * LL

        Parameters:
        X       -- A np.array. Each array in X needs to be of shape (-1,1)
        """
        _, log_likelihood, _ = self.expectation(X)
        return -1 * log_likelihood

    def update_complexity(self):
        """
        Update number of Gaussians for cross-validation.
        """
        self.k += 1

    def clear(self):
        """
        Clear model parameters for cross-validation.
        """
        self.gaussians = []
        self.weights = []
