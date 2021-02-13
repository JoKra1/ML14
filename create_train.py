from encoder import Encoder
import numpy as np
from scratch_builds.pca import PCA
import pandas as pd
from matplotlib import pyplot as plt




if __name__ == "__main__":

    train = "data/fake_news/train.csv"
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
    data = data.to_numpy()
    print(data.shape)

    pca = PCA(500)
    pca.fit(data)
    features = pca.encode(data)
    np.savetxt("data/pca_reduced.csv",features,delimiter=",")

    ### Attach PCA data ###
    data = pd.read_csv("data/pca_reduced.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    e.data["pca"] = data

    ### Export for training ###
    candidates = ["ids","n_quotes","grammar_ratio","pca","labels"]

    e.generate_combined_training_data(candidates,export="data/train_c_red.csv")

    ### Unigram stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_stem.csv",stem=True,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 1), max_features=10000,export="data/vectorized.csv")

    
    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)

    pca = PCA(500)
    pca.fit(data)
    features = pca.encode(data)
    np.savetxt("data/pca_reduced.csv",features,delimiter=",")

    ### Attach PCA data ###
    data = pd.read_csv("data/pca_reduced.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    e.data["pca"] = data

    ### Export for training ###
    candidates = ["ids","n_quotes","grammar_ratio","pca","labels"]

    e.generate_combined_training_data(candidates,export="data/train_c_red_stem.csv")

    ### Bigram not stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_bigram.csv",stem=False,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 2), max_features=10000,export="data/vectorized.csv")

    
    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    pca = PCA(500)
    pca.fit(data)
    features = pca.encode(data)
    np.savetxt("data/pca_reduced.csv",features,delimiter=",")

    ### Attach PCA data ###
    data = pd.read_csv("data/pca_reduced.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    e.data["pca"] = data

    ### Export for training ###
    candidates = ["ids","n_quotes","grammar_ratio","pca","labels"]

    e.generate_combined_training_data(candidates,export="data/train_c_red_bigram.csv")
    
    
    ### Bigramm stemmed ###

    ### Encode ###
    e = Encoder(train)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train_bigram_stem.csv",stem=True,min_len=10,g_cut=0.1)

    ### TFIDF ###
    e.tfidf_vectorize((1, 2), max_features=10000,export="data/vectorized.csv")

    
    ### PCA ###
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    pca = PCA(500)
    pca.fit(data)
    features = pca.encode(data)
    np.savetxt("data/pca_reduced.csv",features,delimiter=",")

    ### Attach PCA data ###
    data = pd.read_csv("data/pca_reduced.csv",delimiter=",",header=None)
    data = data.to_numpy()
    print(data.shape)
    e.data["pca"] = data

    ### Export for training ###
    candidates = ["ids","n_quotes","grammar_ratio","pca","labels"]

    e.generate_combined_training_data(candidates,export="data/train_c_red_bigram_stem.csv")
    
    