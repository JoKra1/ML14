from encoder import Encoder


if __name__ == "__main__":
    train = "data/fake_news/fake_news/train.csv"
    replacement_pairs = [['“', '"'], ['”', '"'],
                         ['’', "'"], ['—', "-"]]

    ### Encode ###
    e = Encoder(train)
    ### Preprocess ###
    e.preprocess(replacement_pairs,
                 export="data/fake_news/cleaned_train.csv")
    ### TFIDF ###
    e.tfidf_vectorize((1, 1), max_features=10000)
    e.reduce_dim('vectorized', export="data/fake_news/tfidf_reduced.csv")
    ### W2V ###
    e.padd_shrink_documents(maxlen=500)
    e.w2v_embed()
    e.w2v_fit_iter(1000, export="data/fake_news/fake_news/")
