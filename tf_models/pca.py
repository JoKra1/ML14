from sklearn import decomposition
import numpy as np

class PCA(object):
    """Not much more than some wrappers around scikit-learn's PCA.
    
    Parameters:
        n    -- The number of components kept after principal component analysis
    """
    
    def __init__(self, n):
        self.n = n
        self.pca = decomposition.PCA(self.n)
    
    def fit(self, X, epochs=None):
        """Fits the PCA on the data (but does not transform it). Note that epochs is ignored.
        
        Parameters:
            X        -- The data the PCA is fitted to
            epochs   -- Ignored (present to ensure compatibility with cross validation)
        """
        return self.pca.fit(X)
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def calculate_loss(self, X):
        return -1 * self.pca.score(X)
    
    def clear(self):
        self.pca = decomposition.PCA(self.n)
        
    def update_complexity(self):
        self.n += 1
        self.pca = decomposition.PCA(self.n)
    