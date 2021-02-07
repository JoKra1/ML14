from tf_models.random_forest import ForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv

"""
Extract PCA data for visualization in R.
"""

if __name__ == "__main__":
    data = pd.read_csv("data/train_c_red_bigram_stem.csv",delimiter=",",header=None)
  
    data = data.to_numpy()

    X = data[:,0:6]
    Y = data[:,-1]

    with open("data/pca_viz_red_bigram_stem.csv","w",newline="") as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=",")
        csvwriter.writerow(["ID","Nquotes","grammar_r","pc1","pc2","pc3","label"])
        for x,y in zip(X,Y):
            row = list(x)
            row.append(y)
            csvwriter.writerow(row)
