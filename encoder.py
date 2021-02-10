import csv
from spellchecker import SpellChecker
from nltk import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import warnings
import json
import random
import io
import itertools
import gc
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from tf_helpers.generate_training_w2v import generate_training_data
from tf_models.mog import MOG
from tf_models.pca import PCA
from scratch_builds.cross_validate import cross_validate


AUTOTUNE = tf.data.experimental.AUTOTUNE


class Encoder(object):
    """
    Encoder. Allows to pre-process data, calculate a vocabulary based on the
    data, transform all data into tfidf vectors (e.g. one vector of vocabulary dims
    per article), and to learn word embeddings using a word2vec skipgram model.

    ToDo:
    -  Use vectorizer attached to instance to process new (test) input
       (e.g. use vocabulary and TF*IDF vectorizer created on trainigs data
       to transform a test document to a TF*IDF vector)
    - same for W2V: use learned embedding matrix to transform training and test
      documents to vectors of embedding vectors.
    - Dimensionality reduction: PCA or Auto-encoders.
    """
    stop_words = stopwords.words('english')

    def __init__(self, data_path, test_data_path=None):
        """
        Constructor

        Parameters:
        data_path       -- path to data as string.
        """
        super().__init__()
        ### Path to data and dict for storage of preprocessing steps ###
        self.path_d = data_path
        self.data = {"ids": [],
                     "labels": [],
                     "n_quotes": [],
                     "grammar_ratio": [],
                     "preprocessed": [],
                     "vectorized": [],
                     "padded_vectors": []}
        
        ### Path for test data if it is given
        if not test_data_path is None:
            self.path_test_d = test_data_path
            
        ### Dict for storage of preprocessing test steps
        self.test_data = {"ids": [],
                          "n_quotes": [],
                          "grammar_ratio": [],
                          "preprocessed": [],
                          "vectorized": [],
                          "padded_vectors": []}
        ### TFIDF data ###
        self.tfidf = None
        self.vocab = None
        self.vocab_inv = None
        self.max_tfidfvec_len = 0
        ### W2V  ###
        self.w2v = None

    def preprocess(self, replacement_pairs, g_cut=0.4, min_len=20, export=False, stem=False, train=True):
        """
        Preprocessing function. Performs spell checking, removes
        token sequences that are too short, and offers an export of the
        cleaned data.

        Parameters:
        g_cut                   -- Spellcheck cut-off used to justify exclusion.
        replacement_pairs       -- A list of lists. Each sub-list contains 'to_be_replaced' and 'replacement'
        min_len                 -- minimum token sequence to be included.
        export                  -- Either False or string to path for export.
        stem                    -- Whether or not to stem the words (bool)
        train                   -- Whether to preprocess the training or testing set (bool)
        """
        
        spellcheck = SpellChecker(language='en')
        
        cleaned = []
        no_text_counter = 0
        spell_check_counter = 0
        too_short_counter = 0
        
        if train:
            data = pd.read_csv(self.path_d, header=0)
            print(data.head())

            ids = np.array(data["id"])
            texts = np.array(data["text"])

            labels = np.array(data["label"])

            for id, text, label in tqdm(zip(ids, texts, labels), total=len(ids), desc="Pre-Processing"):
                try:
                    n_quotes, grammar_ratio, tokens = self._process_text(text, 
                                                                         replacement_pairs, 
                                                                         g_cut, 
                                                                         min_len, 
                                                                         stem,
                                                                         spellcheck)

                    self.data["ids"].append(id)
                    self.data["labels"].append(label)
                    self.data["n_quotes"].append(n_quotes)
                    self.data["grammar_ratio"].append(grammar_ratio)
                    self.data["preprocessed"].append(tokens)
                except ValueError as e:
                    msg = str(e)
                    
                    if msg == "This article contains no text":
                        no_text_counter += 1
                    elif msg == "Text too short":
                        too_short_counter += 1
                    elif msg == "Text too ingrammatical":
                        spell_check_counter +=1
                    else:
                        raise e

            print(
                f"Number of documents after pre-processing: {len(self.data['ids'])}")

            print(f"Removed {no_text_counter} entries that did not contain text")
            print(f"Removed {spell_check_counter} entries that had a grammar ratio > {g_cut}")
            print(f"Removed {too_short_counter} entries that had less than {min_len} tokens after removing stopwords")

            if type(export) == str:
                exp = {"ids": self.data["ids"],
                       "labels": self.data["labels"],
                       "n_quotes": self.data["n_quotes"],
                       "grammar_ratio": self.data["grammar_ratio"],
                       "preprocessed": self.data["preprocessed"]
                       }
                cleaned = pd.DataFrame.from_dict(exp)
                cleaned.to_csv(export, sep=",", index=False,encoding="utf-8")
        else:
            data = pd.read_csv(self.path_test_d, header=0)
            print(data.head())

            ids = np.array(data["id"])
            texts = np.array(data["text"])

            for id, text in tqdm(zip(ids, texts), total=len(ids), desc="Pre-Processing"):
                try:
                    n_quotes, grammar_ratio, tokens = self._process_text(text, 
                                                                         replacement_pairs, 
                                                                         g_cut, 
                                                                         min_len, 
                                                                         stem,
                                                                         spellcheck)
                    
                    self.test_data["ids"].append(id)
                    self.test_data["n_quotes"].append(n_quotes)
                    self.test_data["grammar_ratio"].append(grammar_ratio)
                    self.test_data["preprocessed"].append(tokens)
                except ValueError as e:
                    msg = str(e)
                    
                    if msg == "This article contains no text":
                        no_text_counter += 1
                    elif msg == "Text too short":
                        too_short_counter += 1
                    elif msg == "Text too ingrammatical":
                        spell_check_counter +=1
                    else:
                        raise e
                        
                    continue

                

            print(
                f"Number of documents after pre-processing: {len(self.test_data['ids'])}")

            print(f"Removed {no_text_counter} entries that did not contain text")
            print(f"Removed {spell_check_counter} entries that had a grammar ratio > {g_cut}")
            print(f"Removed {too_short_counter} entries that had less than {min_len} tokens after removing stopwords")

            if type(export) == str:
                exp = {"ids": self.test_data["ids"],
                       "n_quotes": self.test_data["n_quotes"],
                       "grammar_ratio": self.test_data["grammar_ratio"],
                       "preprocessed": self.test_data["preprocessed"]
                       }
                cleaned = pd.DataFrame.from_dict(exp)
                cleaned.to_csv(export, sep=",", index=False,encoding="utf-8")

    def tfidf_vectorize(self, ngrams, min_df=1, max_features=None, train=True, export=False):
        """
        Creates TF*IDF trainings matrix based on min idf feature selection.
        Also creates a dictionary of the features considered important
        according to this criterion. This vocabulary is also inversed and
        can later be used for word2vec or BERT.

        Details:
        In a first step a vocabulary is created, using the TF of each word
        to determine its importance. The vocabulary size then equals the
        size of the TF*IDF vectors created for each document: each document
        will be represented as a sparse vector that contains the TF*IDF value
        for each word in the vocabulary for that particular document (if included).
        These vectors can be used to train classifiers such as a logistic regression
        model or a SVM. They should however be reduced in dimensionality: If we
        want to keep the most important 10.000 words we will have 10.000 features
        and only 20.000 trainings points. Not so good.

        The vocabulary used here is a requirement for the W2V related preprocessing.
        So this method needs to be called beforehand.

        Parameters:
        ngrams              -- what kind of ngrams to consider, tuple
        min_df              -- min_df for a feature to be included in the end
        max_features        -- maximum size of vocabulary. Inclusion determined using TF.
        train               -- ToDo.
        """

        if train:
            ### Attach a TFIDFVectorizer to this encoder instance ###
            self.tfidf = TfidfVectorizer(lowercase=False,
                                         ngram_range=ngrams,
                                         min_df=min_df,
                                         max_features=max_features,
                                         sublinear_tf=True)

            # For each document create a vector of TF*IDF values for all words in final vocab ###.
            tfidf_vectors = self.tfidf.fit_transform(
                self.data["preprocessed"]).toarray()

            names = self.tfidf.get_feature_names()
            self.max_tfidfvec_len = len(names) + 1
            print(f"TFIDF-determined vocabulary size is: {len(names) + 1}")

            ### Create Vocabulary with padding ###
            vocab = {'': 0}
            index = 1
            for name in names:
                vocab[name] = index
                index += 1
            vocab_inv = {value: key for key, value in vocab.items()}
            self.vocab = vocab
            self.vocab_inv = vocab_inv

            ### Each vec here is a max_tfidfvec_len*1 dim vector for a single document ###
            for vec in tfidf_vectors:
                self.data["vectorized"].append(vec)

            ### Export vectorized data ###
            if type(export) == str:
                np.savetxt(export, tfidf_vectors, delimiter=",")

        else:
            # For each document create a vector of TF*IDF values for all words in final vocab ###.
            tfidf_vectors = self.tfidf.transform(
                self.test_data["preprocessed"]).toarray()
            
            ### Each vec here is a max_tfidfvec_len*1 dim vector for a single document ###
            for vec in tfidf_vectors:
                self.test_data["vectorized"].append(vec)

            ### Export vectorized data ###
            if type(export) == str:
                np.savetxt(export, tfidf_vectors, delimiter=",")

    def _process_text(self, text, replacement_pairs, g_cut, min_len, stem, spellcheck):
        if type(text) != str:
            ### This is no text ###
            raise ValueError("This article contains no text")

        ### Special characters treatment ###
        for pair in replacement_pairs:
            text = re.sub(pair[0], pair[1], text)

        ### Capture quotes ###
        quotes = re.findall(' "[ a-zA-Z0-9,.;:!?\']+"', text)
        n_quotes = len(quotes)

        ### From now on drop all numbers ###
        text = re.sub(r'[0-9]+', "", text)

        ### Convert text to lower and remove all punctuation ###
        lower = text.lower()
        no_punct = "".join(
            [c for c in lower if c not in string.punctuation])

        ### Tokenize ###
        tokens = word_tokenize(no_punct)

        if not len(tokens):
            ### Too short ###
            raise ValueError("Text too short")

        ### Spell check (note this also includes neo-anglicisms) ###
        misspelled = spellcheck.unknown(tokens)
        grammar_ratio = len(misspelled)/len(tokens)

        if grammar_ratio > g_cut:
            ### This is likely not English. ###
            # print(tokens[0:10])
            raise ValueError("Text too ingrammatical")

        ### Remove Stopwords ###
        tokens = [t for t in tokens if t not in self.stop_words]

        if len(tokens) < min_len:
            ### Too short ###
            raise ValueError("Text too short")

        ### Stemm words ###
        if stem:
            ps = SnowballStemmer("english")

            tokens = [ps.stem(t) for t in tokens]

        ### Convert back to string for easier storage ###
        tokens = " ".join(tokens)
        
        return (n_quotes, grammar_ratio, tokens)

    def reduce_dim(self, data, method, max_updates=-1, folds=20, epochs=100, target_dim=None, export=False):
        """
        Attempts to reduce the dimensions of either the tf*idf vectors or w2v vectors using
        PCA or MOGs.

        Parameters:
            data        -- string: either 'vectorized' or w2v
            method      -- which method to use for dimensionality reduction. Current options: 'pca' or 'mog'
            max_updates -- max updates to perform: each increasing model flexibility. (default terminates once test loss increases)
            folds       -- number of folds to create when using CV
            epochs      -- how often to iterate over a fold when using CV
            target_dim  -- optional. If not specified determined using CV, else the target dim
        """

        dat = [d.reshape(-1, 1) for d in self.data[data]]
        dat = np.array(dat).reshape(-1, self.max_tfidfvec_len-1)
        print(dat[:5])
        
        if method == 'pca':
            _Model = PCA
        elif method == 'mog':
            _Model = MOG
        else:
            raise ValueError("method must be one of ['pca', 'mog']")

        if target_dim is None:
            m = _Model(1)
            optimum, optimum_flex, _ = cross_validate(
                [m], dat, None, folds, max_updates, epochs)
            m = _Model(optimum_flex + 1)  # optimum flex starts at 0
            m.fit(dat, epochs)
        else:
            m = _Model(target_dim)
            m.fit(dat, epochs)

        reduced = m.transform(dat)
        self.data["tfidf_reduced"] = reduced
        if type(export) == str:
            np.savetxt(export, reduced, delimiter=",")
    
    def generate_combined_training_data(self, candidates, export=False, train=True):
        """
        Combines all available features into training set
        """
        if train:
            data = [self.data[candidate] for candidate in candidates]
        else:
            data = [self.test_data[candidate] for candidate in candidates]
            
        output = []
        for row in zip(*data):
            new_row = []
            for entry in row:
                if isinstance(entry,(list,np.ndarray)):
                    for i in entry:
                        new_row.append(i)
                else:
                    new_row.append(entry)
            output.append(np.array(new_row))
        output = np.array(output)
        if type(export) == str:
            np.savetxt(export, output, delimiter=",")
        return output
