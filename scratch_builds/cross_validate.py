import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
from tqdm import tqdm
from sklearn.model_selection import KFold
import sys
from sklearn.datasets import load_iris


def cross_validate(models, features, labels, k, updates, epochs, shuffle=False):
    """
    Performs cross-validation for supervised and unsupervised models.
    Assumes that each model is an object with at least the following methods:
        - fit(Data,Epochs) for unsupervised / fit(features,labels,Epochs) for supervised
        - calculate loss(data) for unsupervised / calculate_loss(features,labels) for supervised
        - update_complexity() some function leading to an increase in model flexibility.
        - clear() reset model parameters

    Parameters:
        models          -- list with model objects
        features        -- np.array with features (can again be np.arrays)
        labels          -- np.array with labels
        k               -- number of splits
        updates         -- number of updates to model flexibility (-1 == terminate once test loss increases)
        epochs          -- number of epochs to train at each fold
        shuffle         -- whether to shuffle the data before splitting

    Sources:
    Implementation based on reader in the LN
    """
    print("Starting cross-validation")
    per_model_losses = []
    optimum = sys.maxsize
    optimum_flex = 0
    for model_index, model in enumerate(models):
        kf = KFold(n_splits=k, shuffle=False)
        losses_train = []
        losses_test = []
        iterator = updates if updates != -1 else 10000
        for u in range(iterator):
            train_loss_u = []
            test_loss_u = []
            ### Prepare all folds and iterate over them ###
            for train_indices, test_indices in kf.split(features):
                features_train = features[train_indices]
                features_test = features[test_indices]

                ### Supervised ###
                if not labels is None:
                    labels_train = labels[train_indices]
                    labels_test = labels[test_indices]
                    models.fit(features_train, labels_train, epochs)
                    train_loss = model.calculate_loss(
                        features_train, labels_train)
                    test_loss = model.calculate_loss(
                        features_test, labels_test)
                ### Unsupervised ###
                else:
                    model.fit(features_train, epochs)
                    train_loss = model.calculate_loss(features_train)
                    test_loss = model.calculate_loss(features_test)
                ### Append loss for training and test for each fold ###
                train_loss_u.append(train_loss)
                test_loss_u.append(test_loss)
                model.clear()
            ### Average over all folds for a given model flexibility ###
            avg_train_loss = np.mean(train_loss_u)
            avg_test_loss = np.mean(test_loss_u)

            print(
                f"Model: {model_index + 1} Flexibility score: {u} Train loss: {avg_train_loss} Test loss: {avg_test_loss}")

            if avg_test_loss < optimum:
                optimum = avg_test_loss
                optimum_flex += 1
            elif updates == -1:
                return optimum, optimum_flex, per_model_losses
            losses_train.append(avg_train_loss)
            losses_test.append(avg_test_loss)

            plt.plot(range(len(losses_test)), losses_test, color="orange")
            plt.plot(range(len(losses_train)), losses_train, color="blue")
            plt.show(block=False)
            plt.pause(1.5)
            plt.close()

            ### Increase model flexibility ###
            model.update_complexity()

        ### Append per model history of all losses ###
        per_model_losses.append([losses_train[:], losses_test[:]])

    return optimum, optimum_flex, per_model_losses


# if __name__ == "__main__":
    # X = []
    # for i in range(100):
    #     cond = random.random()
    #     if cond > 0.7:
    #         X.append(np.random.multivariate_normal([3.0,4.0],[[0.45,0],[0,0.75]]).reshape(-1,1))
    #     else:
    #         X.append(np.random.multivariate_normal([2.0,3.0],[[0.65,0],[0,0.15]]).reshape(-1,1))
    # X = np.array(X)

    # mo = MOG(1,2)

    # cross_validate([mo],X,None,5,10,10)

    #iris = load_iris()
    # print(iris)
    #data = iris["data"]
    # print(data)
    #X = np.array([np.array(d).reshape(4,1) for d in data])
    #mo = MOG(1,4)
    # cross_validate([mo],X,None,5,5,10)
