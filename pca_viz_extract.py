import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from encoder import Encoder
from scratch_builds.pca import PCA as PCA_s


"""
Extract PCA data for visualization in R.
"""

if __name__ == "__main__":

    train = "data/fake_news/fake_news/train.csv"
    replacement_pairs = [['“', '"'], ['”', '"'],
                         ['’', "'"], ['—', "-"],
                         ["‹", ""], ["›", ""],
                         ["»", ""], ["‘", ""],
                         ["«", ""], ["■", ""],
                         ["🎉", ""], ["•", ""],
                         ["©", ""], ["🔥", ""],
                         ["😭",""], ["👏", ""],
                         ["😈", ""], ["❤", ""],
                         ["😱", ""], ["😝", ""],
                         ["🎉", ""], ["💋", ""],
                         ["🎈", ""], ["🍾", ""],
                         ["👊🏾", ""], ["🏃‍♀️", ""],
                         ["🍑", ""], ["🧀", ""],
                         ["®", ""], ["👇", ""],
                         ["😂", ""], ["❤️", ""],
                         ["😬", ""], ["🤔", ""],
                         ["😉", ""], ["🐸", ""],
                         ["😡", ""], ["😳", ""],
                         ["🙂", ""], ["💔", ""],
                         ["﻿", ""], ["😎", ""],
                         ["…🕶…", ""], ["👍", ""]]

    ### Unigram not stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train.csv",stem=False,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 1), max_features=10000,export="data/vectorized.csv")

    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    pca = PCA_s(10000)
    pca.fit(data)
    singular = pca.export_eigen_values()
    np.savetxt("data/pca_singular.csv",singular,delimiter=",")

    ### Unigram stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_stem.csv",stem=True,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 1), max_features=10000,export="data/vectorized.csv")

    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    pca = PCA_s(10000)
    pca.fit(data)
    singular = pca.export_eigen_values()
    np.savetxt("data/pca_singular_stemmed.csv",singular,delimiter=",")

    ### Bigram not stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_bigram.csv",stem=False,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 2), max_features=10000,export="data/vectorized.csv")

    
    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    pca = PCA_s(10000)
    pca.fit(data)
    singular = pca.export_eigen_values()
    np.savetxt("data/pca_singular_bigram.csv",singular,delimiter=",")
    
    
    ### Bigramm stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_bigram_stem.csv",stem=True,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 2), max_features=10000,export="data/vectorized.csv")

    
    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    pca = PCA_s(10000)
    pca.fit(data)
    singular = pca.export_eigen_values()
    np.savetxt("data/pca_singular_bigram_stemmed.csv",singular,delimiter=",")


    ### Collect individual training files for visualization ###
    paths = ["data/train_c_red.csv", "data/train_c_red_stem.csv",
             "data/train_c_red_bigram.csv", "data/train_c_red_bigram_stem.csv"]
    file_endings = ["", "_stem", "_bigram", "_bigram_stem"]

    for path, ending in zip(paths,file_endings):
        data = pd.read_csv(path,delimiter=",",header=None)
    
        data = data.to_numpy()

        X = data[:,0:8]
        Y = data[:,-1]

        with open("data/pca_viz" + ending + ".csv","w",newline="") as csvfile:
            csvwriter = csv.writer(csvfile,delimiter=",")
            csvwriter.writerow(["ID","Nquotes","grammar_r","pc1","pc2","pc3","pc4","pc5","label"])
            for x,y in zip(X,Y):
                row = list(x)
                row.append(y)
                csvwriter.writerow(row)
