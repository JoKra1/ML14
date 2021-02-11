from tf_models.random_forest import ForestClassifier
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv
from tqdm import tqdm
from sklearn.model_selection import KFold
import math


"""
Cross-validation for random forest
"""

if __name__ == "__main__":
    # Styling
    matplotlib.rcParams.update({'font.size': 15})
    ### Load in data ###
    data = pd.read_csv("data/fake_news/train_c_red.csv",delimiter=",",header=None)
    print(data.head)
    data = data.to_numpy()

    ### Prepare training data ###
    X = data[:,1:103]
    Y = data[:,-1]

    feature_names = ["n_quotes","grammar_r."]
    for i in range(100):
        feature_names.append(f"pc{i+1}")
    
    forest = ForestClassifier(2500,"entropy",max_features=8)
    forest.fit(X,Y)

    importance_features = []
    for name, importance in zip(feature_names[0:20],forest.forest.feature_importances_[0:20]):
        importance_features.append(importance)
        print(f"Feature: {name} Importance: {importance}")
    
    plt.bar(range(20),importance_features)
    plt.xticks(range(20),feature_names[0:20],rotation="vertical")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importance for first 20 features")

    plt.show()

    ### setup cross-validation ###
    kf = KFold(n_splits=10,shuffle=False)
    acc = []
    conf_matrix = np.zeros((2,2))
    for train_indices, test_indices in kf.split(X):
        ### Split fold ###
        train_X = X[train_indices]
        train_Y = Y[train_indices]

        test_X = X[test_indices]
        test_Y = Y[test_indices]


        forest = ForestClassifier(2500,"entropy",max_features=8)
        forest.fit(train_X,train_Y)
        val_acc = forest.forest.score(test_X,test_Y)
        forest.plot_estimator(0,feature_names)
        conf_fold = forest.print_confusion_matrix(test_X,test_Y)
        conf_matrix += conf_fold
        acc.append(val_acc)
    
    print("Final Confusion matrix:")
    print(conf_matrix)
    plt.bar(range(1,11),acc)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy on validation set")
    plt.title("10 - Fold: Random Forest")
    plt.show()

    # Mean + SD
    mu = sum(acc)/10
    print(f"Mean: {mu}")
    ss = [(acc_i - mu)**2 for acc_i in acc]
    ms = sum(ss)/9 # Variance estimate so using n-1!
    sd = math.sqrt(ms)
    print(f"SD: {sd}")









