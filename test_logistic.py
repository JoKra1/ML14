import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read pre-processed training data
dat_train = pd.read_csv("data/fake_news/train_c_red_bigram.csv", header=None)

labels = dat_train.iloc[:, -1]
labels = labels.to_numpy()

dat_train = dat_train.iloc[:, 1:-1]
dat_train = dat_train.to_numpy()

# Read pre-processed testing data
dat = pd.read_csv("data/fake_news/test_c_red_bigram.csv", header=None)

ids = dat.iloc[:, 0]
dat = dat.iloc[:, 1:]

# Get all testing IDs so we know what's been discarded during pre-processing
all_ids = pd.read_csv("data/fake_news/test.csv")['id']

# Train the classifier
clf = LogisticRegression(penalty="l2", solver="liblinear", dual=False, C=18)
clf.fit(dat_train, labels)

# Test the classifier
labs = clf.predict(dat)

# Classify text as unreliable if not in pre-processed testing data
res = dict([(id, lab) for id, lab in zip(ids, labs)])

for id in all_ids:
    if not id in res.keys():
        res[id] = 1 # 1 == unreliable
        
res = [(id, lab) for (id, lab) in res.items()]

# Place results in DataFrame
results = pd.DataFrame(res, columns=["id", "label"], dtype=int)
results = results.set_index("id")
results = results.sort_values("id")

# Write results to file
results.to_csv("data/logistic_submission.csv")