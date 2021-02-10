import numpy as np
from sklearn.decomposition import PCA as skPCA




class PCA(object):
    """
    Performs PCA on the covariance matrix of the centered training data using singular value decomposition.
    """

    def __init__(self,m):
        self.m = m
        self.pca = skPCA(m,random_state=0,svd_solver="full")
    
    def calculate_dissimilarity(self,m):
        """
        Parameters:
        m       -- target dim
        """

        print(f"Variance explained for m = {m} is: {sum(self.pca.explained_variance_ratio_)}")

        return 
    
    def export_eigen_values(self):
        return self.pca.explained_variance_
    
    def encode(self,X):
        """
        Converts a numpy array into the encoded representation using
        only the retained Principal components.
        """
        features = self.pca.transform(X)
        return features
        


    def update_complexity(self,recalculate=False):
        """
        Update number of principal components retained.
        """
        self.m += 1
        if recalculate:
            self.pca = skPCA(self.m,svd_solver="full")
        

    def clear(self):
        """
        Clear model parameters for cross-validation.
        """
        pass

    def fit(self,data):
        """
        Data needs to be of dimension n = samples, m = features
        """
        self.pca.fit(data)

        
        


        