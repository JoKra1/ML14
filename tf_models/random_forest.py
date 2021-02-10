import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import os


class ForestClassifier(object):
    """
    Sources:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    Hands on Machine Learning with SCIKIT-LEARN, Keras, and Tensorflow (Second edition) by Aurelien Geron
    """

    def __init__(self,n,criterion,max_features,bootstrap=True,random_state=0,oob=True):
        ### Attach Forest ###
        self.forest = RandomForestClassifier(criterion=criterion,
                                             random_state=random_state,
                                             max_features=max_features,
                                             bootstrap=bootstrap,
                                             n_estimators=n,
                                             n_jobs=4,
                                             oob_score=oob
                                        )

    def fit(self,X,Y):
        """
        Fit ensemble!
        """
        self.forest.fit(X,Y)
    
    def out_of_bag_estimate(self):
        """
        Return OOB Score
        """
        return self.forest.oob_score_
    
    def estimate_strength(self,x,y):
        """
        An estimate of the strength of the forest based on the OOB estimates
        """
        estimators = self.forest.estimators_
        proportions = np.zeros((len(y),2))
        for estimator in estimators:
            predictions = estimator.predict(x)
            for index, prediction in enumerate(predictions):
                proportions[index,int(prediction)] += 1
        proportions = proportions / len(estimators)

        s = 0
        for index, label in enumerate(y):
            si = proportions[index, int(label)] - proportions[index, int(1-label)]
            s += si
        s = s / len(y)
        print(s)

        print(self.forest.oob_decision_function_)
        return(s)

    def plot_estimator(self,index,names):
        """
        Plot an individual estimator
        """
        estimator = self.forest.estimators_[index]
        export_graphviz(estimator,
                        feature_names=names,
                        out_file="tree.dot",
                        class_names=["Reliable","Unreliable"],
                        rounded=True,
                        filled=True)
        os.system('dot -Tpng tree.dot -o tree.png')
    
    def print_confusion_matrix(self,x,y):
        """
        Plot confusion matrix
        """
        y_hat = self.forest.predict(x)
        matr = confusion_matrix(y_true=y,y_pred=y_hat,labels=[0,1])
        print("Confusion matrix:\n")
        print(matr)
        return matr



        
    
        
