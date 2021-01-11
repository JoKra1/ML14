from sklearn.mixture import GaussianMixture
import numpy as np
import random


class MOG(object):
    """
    Fast implementation of MOG model. See scratch build folder for discussion
    """

    def __init__(self, k, cov_type='diag'):
        self.k = k
        self.cov_type = cov_type
        self.mixture = GaussianMixture(
            n_components=k, random_state=0, covariance_type=cov_type)

    def fit(self, X, epochs):
        self.mixture.max_iter = epochs
        self.mixture.fit(X)

    def calculate_loss(self, X):

        return -1 * self.mixture.score(X)

    def expectation(self, X):
        prob_matrix = self.mixture.predict_proba(X)
        return prob_matrix
    
    def transform(self, X):
        return expectation(self, X)

    def clear(self):
        self.mixture = GaussianMixture(
            n_components=self.k, random_state=0, covariance_type=self.cov_type)

    def update_complexity(self):
        self.k += 1
        self.mixture = GaussianMixture(
            n_components=self.k, random_state=0, covariance_type=self.cov_type)
