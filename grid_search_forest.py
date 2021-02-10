from tf_models.random_forest import ForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv

"""
Grid Search for random forest. Repeated fpr bigram and unigram data!
"""

if __name__ == "__main__":
    paths = ["data/train_c_red.csv", "data/train_c_red_stem.csv",
             "data/train_c_red_bigram.csv", "data/train_c_red_bigram_stem.csv"]
    file_endings = ["", "stem", "bigram", "bigram_stem"]

    for path, ending in zip(paths,file_endings):
        data = pd.read_csv(path,delimiter=",",header=None)
        print(data.head)
        data = data.to_numpy()

        pcs = [50,75,100,150]
        for pc in pcs:
            print(f"Principal components retained: {pc}")
            upper = pc + 3
            X = data[:,1:upper]
            Y = data[:,-1]


            N = [100,200,300,400,500,1000,1500,2000,2500,3000]
            N_f = [1,2,3,4,5,6,7,8,9,10,20,30,50]
            
            results = []
            with open("data/forest" + str(pc) + ending + ".csv","w",newline="") as csvfile:
                csvwriter = csv.writer(csvfile,delimiter=",")
                csvwriter.writerow(["N","NF","OOB"])
                for n in N:
                    results.append([])
                    for nf in N_f:
                        forest = ForestClassifier(n,"entropy",max_features=nf)
                        forest.fit(X,Y)
                        oob = forest.out_of_bag_estimate()
                        results[-1].append(oob)
                        print(f"{n},{nf},{oob}")
                        csvwriter.writerow([n, nf, oob])

    
    