from tf_models.random_forest import ForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from tqdm import tqdm
from sklearn.model_selection import KFold


"""
Cross-validation for random forest
"""

if __name__ == "__main__":
    ### Load in data ###
    data = pd.read_csv("data/train_c_red.csv",delimiter=",",header=None)
    print(data.head)
    data = data.to_numpy()

    data_stem = pd.read_csv("data/train_c_red_stem.csv",delimiter=",",header=None)
    print(data_stem.head)
    data_stem = data_stem.to_numpy()

    ### Prepare training data for stemmed and unstemmed versions ###
    X = data[:,1:103]
    Y = data[:,-1]

    X_stem = data_stem[:,1:103]
    Y_stem = data_stem[:,-1]

    feature_names = ["n_quotes","grammar_ratio"]
    for i in range(100):
        feature_names.append(f"pc{i+1}")

    ### setup cross-validation ###
    kf = KFold(n_splits=2,shuffle=False)
    acc = []
    for train_indices, test_indices in kf.split(X_stem):
        ### Split fold ###
        train_X = X_stem[train_indices]
        train_Y = Y_stem[train_indices]

        test_X = X_stem[test_indices]
        test_Y = Y_stem[test_indices]


        forest = ForestClassifier(200,"entropy",max_features=1)
        forest.fit(train_X,train_Y)
        val_acc = forest.forest.score(test_X,test_Y)
        forest.plot_estimator(0,feature_names)
        forest.print_confusion_matrix(test_X,test_Y)
        acc.append(val_acc)
    
    plt.plot(range(10),acc)
    plt.show()
    print(sum(acc)/10)









