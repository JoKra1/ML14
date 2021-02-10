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
    data = pd.read_csv("data/fake_news/train_c_red_bigram.csv",delimiter=",",header=None)
    print(data.head)
    data = data.to_numpy()

    data_stem = pd.read_csv("data/fake_news/train_c_red_bigram_stem.csv",delimiter=",",header=None)
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
    
    forest = ForestClassifier(3000,"entropy",max_features=6)
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
    for train_indices, test_indices in kf.split(X_stem):
        ### Split fold ###
        train_X = X_stem[train_indices]
        train_Y = Y_stem[train_indices]

        test_X = X_stem[test_indices]
        test_Y = Y_stem[test_indices]


        forest = ForestClassifier(3000,"entropy",max_features=7)
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
    print(sum(acc)/10)









