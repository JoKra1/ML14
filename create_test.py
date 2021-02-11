from encoder import Encoder
import numpy as np
from scratch_builds.pca import PCA
import pandas as pd
from matplotlib import pyplot as plt

"""
Creates test-set training data for logistic regression and random forest.
For logistic regression this creates a dataset with unigrams and bi-grams that is not stemmed.
For the forest this creates a dataset with only unigrams that is not stemmed.
"""

if __name__ == "__main__":
    train = "data/fake_news/train.csv"
    test = "data/fake_news/test.csv"
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
    e = Encoder(train,test_data_path=test)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train.csv",stem=False,min_len=10,g_cut=0.1)
    e.preprocess(replacement_pairs, export="data/cleaned_test.csv",stem=False,min_len=10,g_cut=0.1,train=False)

    ### TFIDF ###
    e.tfidf_vectorize((1, 1), max_features=10000,export="data/vectorized.csv")
    e.tfidf_vectorize((1, 1), export="data/vectorized_test.csv",train=False)

    ### PCA ###

    ### Train ###
    data_train = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    data_train = data_train.to_numpy()
    print(data_train.shape)
    pca = PCA(500)
    pca.fit(data_train)

    ### Test ###
    data_test = pd.read_csv("data/vectorized_test.csv",delimiter=",",header=None)
    data_test = data_test.to_numpy()
    print(data_test.shape)
    features = pca.encode(data_test)
    e.test_data["pca"] = features

    candidates = ["ids","n_quotes","grammar_ratio","pca"]

    e.generate_combined_training_data(candidates,export="data/test_c_red.csv",train=False)


    ### Unigram + Bigram not stemmed ###

    ### Encode ###
    e = Encoder(train,test_data_path=test)

    ### Preprocess ###
    e.preprocess(replacement_pairs, export="data/cleaned_train.csv",stem=False,min_len=10,g_cut=0.1)
    e.preprocess(replacement_pairs, export="data/cleaned_test_bigram.csv",stem=False,min_len=10,g_cut=0.1,train=False)

    ### TFIDF ###
    e.tfidf_vectorize((1, 2), max_features=10000,export="data/vectorized_bigram.csv")
    e.tfidf_vectorize((1, 2), export="data/vectorized_test_bigram.csv",train=False)

    ### PCA ###

    ### Train ###
    data_train = pd.read_csv("data/vectorized_bigram.csv",delimiter=",",header=None)
    data_train = data_train.to_numpy()
    print(data_train.shape)
    pca = PCA(500)
    pca.fit(data_train)

    ### Test ###
    data_test = pd.read_csv("data/vectorized_test_bigram.csv",delimiter=",",header=None)
    data_test = data_test.to_numpy()
    print(data_test.shape)
    features = pca.encode(data_test)
    e.test_data["pca"] = features

    candidates = ["ids","n_quotes","grammar_ratio","pca"]

    e.generate_combined_training_data(candidates,export="data/test_c_red_bigram.csv",train=False)