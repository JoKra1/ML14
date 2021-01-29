import numpy as np




class PCA(object):
    """
    Performs PCA on the covariance matrix of the centered training data using singular value decomposition.
    """

    def __init__(self,m):
        self.m = m
        self.mu = None
        self.U = None
        self.S = None
        self.red_U = None
        self.red_S = None
    
    def calculate_dissimilarity(self,m,S):
        """
        Parameters:
            m       -- target dim
            S       -- S part of SVD (contains eigen values)
        """
        relative_dissimilarity = sum(S[m:])/sum(S)
        print(f"Relative dissimilarity for m = {m} is: {relative_dissimilarity}")
        return relative_dissimilarity

    def calculate_mean_square(self,m,S):
        """
        Parameters:
            m       -- target dim
            S       -- S part of SVD (contains eigen values)
        """
        msq = sum(S[m:])
        print(f"Mean-Square for m = {m} is: {msq}")
        return msq
    
    def calculate_stats(self,m):
        """
        Parameters:
            m       -- target dim
        """
        ### Extract fit statistics ###
        msq = self.calculate_mean_square(m,self.S)
        rel_diss = self.calculate_dissimilarity(m,self.S)
        return msq,rel_diss
    
    def calculate_reduced(self,m):
        """
        Parameters:
            m       -- target dim
        """
        self.red_S = self.S[:m]
        self.red_U = self.U[:,:m]


    def calculate_loss(self, X):
        """
        Returns quadratic loss for encoding decoding on test/train data in X.
        """
        loss = 0
        N = X.shape[1]
        for i in range(X.shape[1]):
            x = X[:,i] - self.mu

            feature = self.red_U.T.dot(x)
            decoded = self.mu + self.red_U.dot(feature)
            
            loss += sum((x - decoded)**2)
        loss = 1/N * loss
        print(f"Loss at m = {self.m}: {loss}")
        return loss

    def update_complexity(self,recalculate=False):
        """
        Update number of principal components retained.
        """
        self.m += 1
        if recalculate:
            self.calculate_reduced(self.m)
        

    def clear(self):
        """
        Clear model parameters for cross-validation.
        """
        pass

    def fit(self,data):
        ### Number of training points ###
        N = data.shape[1]

        ### Transpose for mean and centering operations ###
        data = data.T

        ### Get mean ###
        mean_dat = np.mean(data,axis=0)

        ### Center data ###
        centered = data - mean_dat

        ### Transpose again to match Reader's n by N dimensionality ###
        centered = centered.T

        ### Calculate covariance matrix of input vectors ###
        COV = 1/N * centered.dot(centered.T) # n by n matrix where n is number of features/vocabs

        ### SVD, U has eigen vectors while S has corresponding eigen values on diagonal ###
        U, S, V = np.linalg.svd(COV)
        self.U = U
        self.S = S
        self.mu = mean_dat

        ### Obtain reduced eigen vector matrix and eigen value list ###
        self.calculate_reduced(self.m)

        
        


        