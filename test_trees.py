from tf_models.random_forest import ForestClassifier
import pandas as pd
import numpy as np
import csv


if __name__ == "__main__":
    ### Load in train data ###
    data_train = pd.read_csv("data/fake_news/train_c_red.csv",delimiter=",",header=None)
    data_train = data_train.to_numpy()

    ### Prepare train data ###
    X_train = data_train[:,1:103]
    Y_train = data_train[:,-1]

    forest = ForestClassifier(2500,"entropy",max_features=8)
    forest.fit(X_train,Y_train)

    ### Load in test data ###
    data_test = pd.read_csv("data/fake_news/test_c_red.csv",delimiter=",",header=None)
    data_test = data_test.to_numpy()

    ### Prepare test data ###
    X_test = data_test[:,1:103]

    ID_predictions = data_test[:,0] # Collect IDS for which to make predictions

    ### Obtain predictions ###
    predictions_test = forest.forest.predict(X_test)

    ### Collect predictions and apply heuristic ###
    id_lab_iter = zip(data_test[:,0],predictions_test)

    original_test = pd.read_csv("data/fake_news/test.csv",delimiter=",",header=0)
    original_test = original_test.to_numpy()

    ids = []
    labels = []
    for id_o in original_test[:,0]:
        if id_o in ID_predictions:
            id_pred,label_pred = next(id_lab_iter)
            ids.append(id_pred)
            labels.append(label_pred)
        else:
            ids.append(id_o)
            labels.append(1) # Heuristic discussed in report
    
    # Save final submission file for trees
    with open("data/tree_submission.csv","w",newline="") as csvfile:
        csv_writer = csv.writer(csvfile,delimiter=",")
        csv_writer.writerow(["ID","Label"])
        for id_pred,label_pred in zip(ids,labels):
            csv_writer.writerow([int(id_pred),int(label_pred)])
    
    



