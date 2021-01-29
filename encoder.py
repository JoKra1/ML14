import csv
from spellchecker import SpellChecker
from nltk import word_tokenize
import re
import nltk
from nltk.corpus import stopwords
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
from tf_models.w2v import Word2Vec
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

    def __init__(self, data_path):
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
        ### TFIDF data ###
        self.tfidf = None
        self.vocab = None
        self.vocab_inv = None
        self.max_tfidfvec_len = 0
        ### W2V  ###
        self.w2v = None

    def preprocess(self, replacement_pairs, g_cut=0.4, min_len=20, export=False):
        """
        Preprocessing function. Performs spell checking, removes
        token sequences that are too short, and offers an export of the
        cleaned data.

        Parameters:
        g_cut                   -- Spellcheck cut-off used to justify exclusion.
        replacement_pairs       -- A list of lists. Each sub-list contains 'to_be_replaced' and 'replacement'
        min_len                 -- minimum token sequence to be included.
        export                  -- Either False or string to path for export.     
        """

        data = pd.read_csv(self.path_d, header=0)
        print(data.head())

        ids = np.array(data["id"])
        texts = np.array(data["text"])
        labels = np.array(data["label"])

        spellcheck = SpellChecker(language='en')

        cleaned = []
        for id, text, label in tqdm(zip(ids, texts, labels), total=len(ids), desc="Pre-Processing"):

            if type(text) != str:
                ### This is no text ###
                continue

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
                continue

            ### Spell check (note this also includes neo-anglicisms) ###
            misspelled = spellcheck.unknown(tokens)
            grammar_ratio = len(misspelled)/len(tokens)

            if grammar_ratio > g_cut:
                ### This is likely not English. ###
                continue

            ### Remove Stopwords ###
            tokens = [t for t in tokens if t not in self.stop_words]

            if len(tokens) < min_len:
                continue

            ### Convert back to string for easier storage ###
            tokens = " ".join(tokens)

            self.data["ids"].append(id)
            self.data["labels"].append(label)
            self.data["n_quotes"].append(n_quotes)
            self.data["grammar_ratio"].append(grammar_ratio)
            self.data["preprocessed"].append(tokens)

        print(
            f"Number of documents after pre-processing: {len(self.data['ids'])}")

        if type(export) == str:
            exp = {"ids": self.data["ids"],
                   "labels": self.data["labels"],
                   "n_quotes": self.data["n_quotes"],
                   "grammar_ratio": self.data["grammar_ratio"],
                   "preprocessed": self.data["preprocessed"]
                   }
            cleaned = pd.DataFrame.from_dict(exp)
            cleaned.to_csv(export, sep=",", index=False)

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

            ### Each vec here is a max_tfidfvec_len*1 dim vector for a sigle document ###
            for vec in tfidf_vectors:
                self.data["vectorized"].append(vec)

            ### Export vectorized data ###
            if type(export) == str:
                np.savetxt("data/fake_news/vectorized.csv",tfidf_vectors,delimiter=",")

        else:
            ### ToDo: Use vectorizer attached to instance to process new (test) input ###
            pass

    def padd_shrink_documents(self, maxlen=None, export=False, import_text=False):
        """
        Based on idf vocab generated earlier shrink size
        of each document. Then calculate max document length (if not
        specified) and padd all shorter/truncate all longer documents
        to match length. Then vectorize documents based on dictionary
        created earlier. Allows to export-import the word-reduced documents
        (see details).

        Details:
        To train the W2V model we need to represent each documents using tokens (
        unique integer for each word in the document). The vectorized documents
        are to be padded with zeros if they are shorter than the max length (either
        a specified value or determined by the longest document). If we truncate
        the max-length to say 300, semantic structure only observable in later parts of
        longer documents will not be encoded in the final word embeddings learned by
        W2V (if there is such semantic sturcture restircted to later sections of
        longer articles is something I do not know). In a first step the method here
        reduces each article to only contain the words included in the vocabulary generated
        earlier. In a second step it pads each document with a '' (padding char) until
        it matches the max length determined or truncates a document that exceeds the
        max-length. Finally, it uses the vocabulary dictionary created by the
        tfidf_vectorize() method to convert the word reduced and padded/truncated documents
        in text-form to token sequences (unique integer for each word.).

        Parameters:
        maxlen              -- max lenght of a document to keep for W2V training
        export              -- either boolean or string to path for export
        import_text         -- either boolean or string to path for import of word reduced documents.

        """
        ### Collect documents and split them ###
        docs = [p.split(" ") for p in self.data["preprocessed"]]
        names = self.tfidf.get_feature_names()

        ### Import or calculate reduced documents ###
        if type(import_text) != str:
            ### Reduce documents to include only word in vocabulary created earlier. ###
            warnings.warn(
                "Depending on min IDF cut-off selected earlier this may take a while. Consider importing the reduced documents.")
            docs = [[d_idf for d_idf in d if d_idf in names] for d in tqdm(
                docs, total=len(docs), desc="Reducing document size")]
            if type(export) == str:
                """
                Export only reduced text.
                """
                docs_exp = [" ".join(doc) for doc in docs]
                exp = pd.DataFrame(docs_exp, columns=["reduced_text"])
                exp.to_csv(export, sep=",", index=False)

        else:
            docs = pd.read_csv(import_text, header=0).values.tolist()
            try:
                docs = [d[0].split(" ") for d in docs]
            except AttributeError:
                warnings.warn(
                    "Empty documents in import. Consider increasing Vocabulary sice.")
                docs = [d[0].split(" ") if type(d[0]) ==
                        str else [''] for d in docs]

        ### Collect length of each shrinked doc and calc max_len###
        if not maxlen:
            docs_l = [len(d) for d in docs]
            max_len = max(docs_l)
        else:
            max_len = maxlen

        print(f"Max document length was: {max_len}")

        ### Append vectorized padded doc ###
        data_index = 0
        print("Some example padded vectors + conversions:")
        for d in docs:
            if len(d) > max_len:
                padded_doc = d[:max_len]
            else:
                padded_doc = d
                while len(padded_doc) < max_len:
                    padded_doc.append('')
            vect_pad_doc = [self.vocab[s] for s in padded_doc]
            self.data["padded_vectors"].append(np.array(vect_pad_doc))
            if data_index < 5:
                print(
                    f"Vectorized padded : {vect_pad_doc[:10]} => {padded_doc[:10]}")
            data_index += 1

    def sample_negative_w2v(self, sequences, window_n=4, negative_n=2, seed=42):
        """
        'Generates skip-gram pairs with negative sampling for a list of sequences
        (int-encoded sentences) based on window size, number of negative samples
        and vocabulary size'. Utilizes negative sampling as discussed by Mikolov. et al. (2013).

        Details:
        See https://www.tensorflow.org/tutorials/text/word2vec

        Parameters:
        sequences:      	-- a list of documents, can also be entire corpus
        window_n:           -- window size for context creation
        negative_n          -- number of negative samples
        seed                -- rng seed

        Taken & adapted from:
        https://www.tensorflow.org/tutorials/text/word2vec
        """

        print(
            f"Creating training samples for W2V based on {len(sequences)} documents")
        targets, contexts, labels = generate_training_data(sequences=sequences,
                                                           window_size=window_n,
                                                           num_ns=negative_n,
                                                           vocab_size=self.max_tfidfvec_len,
                                                           seed=seed)

        print(f"Number of training points for w2v model: {len(targets)}")
        return targets, contexts, labels

    def w2v_embed(self, embedding_dim=128, negative_n=2):
        """
        Attaches to this encoder instance, a w2v skip-gram model.

        Details:
        See https://www.tensorflow.org/tutorials/text/word2vec

        Parameters:
        embedding_dim       -- target dimension of semantic vectors
        negative_n          -- number of negative samples. Must match
                               argument passed to sample_negative_w2v()

        Taken & adapted from:
        https://www.tensorflow.org/tutorials/text/word2vec
        """

        self.w2v = Word2Vec(self.max_tfidfvec_len, embedding_dim, negative_n)
        self.w2v.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])

        print("Attached W2V-Skipgram model")

    def w2v_fit(self, batch_size=1024, buffer_size=10000, embedding_dim=128, negative_n=2, epochs=40):
        """
        Fits thw W2V skipgram model on the entire corpus, in one go. Note that this only
        ever works if you have a huge amount of ram or if you selected a relatively short
        max sequence length in the earlier steps (100-200). If you want to train on the
        entire possible sequence length use w2v_fit_iter instead.

        Details:
        See https://www.tensorflow.org/tutorials/text/word2vec

        Parameters:
        batch_size      -- batch size used in one gradient iteration
        buffer_size     -- buffer size used for dataset generation by TF
        embedding_dim   -- target dimension of semantic vectors
        negative_n      -- number of negative samples
        epochs          -- How many time model should see entire corpus

        Taken & adapted from:
        https://www.tensorflow.org/tutorials/text/word2vec
        """
        targets, contexts, labels = self.sample_negative_w2v(self.data["padded_vectors"],
                                                             embedding_dim, negative_n)

        dataset = tf.data.Dataset.from_tensor_slices(
            ((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

        self.w2v.fit(dataset, epochs=epochs)

        print("Finished training w2v")

    def w2v_fit_iter(self, docs_n, batch_size=1024, buffer_size=10000, embedding_dim=128, negative_n=2, epochs=30, export=False):
        """
        The iterative W2V fitting method. For each epoch, the entire corpus is first shuffled.
        In a second step, the corpus is seperated into intervals of length docs_n. Training data
        is generated for the interval and send to the model, which performs 1 pass over the training set.
        This is repeated for all intervals, and once the model has performed one pass over all intervals
        it has completed one epoch since it has performed a pass over the entire corpus.

        Details:
        See https://www.tensorflow.org/tutorials/text/word2vec

        Parameters:
        docs_n          -- number of documents from which training examples should be created
        batch_size      -- batch size used in one gradient iteration
        buffer_size     -- buffer size used for dataset generation by TF
        embedding_dim   -- target dimension of semantic vectors
        negative_n      -- number of negative samples
        epochs          -- How many time model should see entire corpus
        export          -- boolean or string with path to export of W2V weights

        Adapted from:
        https://www.tensorflow.org/tutorials/text/word2vec
        """
        completed_intervals = 0
        for e in range(epochs):
            # For every epoch shuffle the dataset
            sequences = self.data["padded_vectors"][:]
            random.shuffle(sequences)
            intervals = [0]
            ### Create intervals (e.g. range of indices of documents) ###
            while intervals[-1] + docs_n < len(sequences):
                intervals.append(intervals[-1] + docs_n)
            ### Go through all document 'batches' ###
            for i in range(1, len(intervals)):
                batch_seq = sequences[intervals[i-1]:intervals[i]]

                targets, contexts, labels = self.sample_negative_w2v(batch_seq,
                                                                     embedding_dim,
                                                                     negative_n)

                dataset = tf.data.Dataset.from_tensor_slices(
                    ((targets, contexts), labels))
                dataset = dataset.shuffle(buffer_size).batch(
                    batch_size, drop_remainder=True)
                dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

                hist = self.w2v.fit(dataset, epochs=1, verbose=0)
                completed_intervals += 1
                if completed_intervals % 5 == 0:
                    loss = hist.history["loss"][-1]
                    acc = hist.history["accuracy"][-1]
                    print(
                        f"Completed Intervals: {completed_intervals} Loss: {loss} Accuracy: {acc}")

                ### Enforce garbage collection cycle on unnecessary (old) trainings structures ###
                del targets
                del contexts
                del labels
                del dataset
                gc.collect()

            loss = hist.history["loss"][-1]
            acc = hist.history["accuracy"][-1]
            print(f"Epoch: {e+1} Loss: {loss} Accuracy: {acc}")

            if type(export) == str:
                ### Export Wv2 vectors and meta data after each epoch. ###
                self.export_w2v(export)

    def export_w2v(self, export):
        """
        Exports W2V vectors and meta data.

        Details:
        See https://www.tensorflow.org/tutorials/text/word2vec

        Parameters:
        export      -- boolean or string with path to export

        Taken & adapted from:
        https://www.tensorflow.org/tutorials/text/word2vec
        """
        weights = self.w2v.get_layer('w2v_embedding').get_weights()[0]
        vocab = self.tfidf.get_feature_names()

        out_v = io.open(export + 'vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open(export + 'metadata.tsv', 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")
        out_v.close()
        out_m.close()

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
