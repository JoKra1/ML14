from sklearn.linear_model import Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import os

from encoder import Encoder


##################### Matplotlib config #####################

mpl.rc_file_defaults()

mpl.rc('font', size=16)

tick_size = 14
mpl.rc('xtick', labelsize=tick_size)
mpl.rc('ytick', labelsize=tick_size)

mpl.rc('axes', titlesize=14)

mpl.rc('legend', fontsize=14)

mpl.rc('figure', figsize=[7.8, 5.2])


##################### Helper functions #####################

def logistic_loss(w, b, x, y, c):
    # First append the value 1 to all rows in x
    x = np.hstack((x, np.full((x.shape[0], 1), fill_value=1)))
    # Then append b to w
    w = np.hstack((w, b))
    
    # Regularisation term
    reg = 0.5 * np.dot(w, w)
    # Log-likelihood
    ll = c * np.sum(np.log(1 + np.exp(-y * np.dot(x, w))))
    
    return reg + ll

def logistic_cv(dat_og, labels, dims, cs):
    train_losses = []
    train_scores = []
    test_losses = []
    test_scores = []

    for dim in dims:
        for c in cs:
            print(f"Dim: {dim}, C: {c}{10*' '}")

            dat = dat_og[:, :dim]

            _train_losses = []
            _train_scores = []
            _test_losses = []
            _test_scores = []

            kf = KFold(n_splits=10, shuffle=False)
            for epoch, (i_train, i_test) in enumerate(kf.split(dat)):
                print(f"{epoch+1}/{kf.get_n_splits()}", end="\r")

                train_dat, train_lab = dat[i_train], labels[i_train]
                test_dat,  test_lab  = dat[i_test],  labels[i_test]

                clf = LogisticRegression(penalty="l2", solver="liblinear", dual=False, C=c)
                clf.fit(train_dat, train_lab)

                _train_losses.append(logistic_loss(clf.coef_[0], 
                                                   clf.intercept_, 
                                                   train_dat, 
                                                   train_lab, 
                                                   clf.C))
                _train_scores.append(clf.score(train_dat, train_lab))

                _test_losses.append(logistic_loss(clf.coef_[0], 
                                                  clf.intercept_, 
                                                  test_dat, 
                                                  test_lab, 
                                                  clf.C))
                _test_scores.append(clf.score(test_dat, test_lab))

            train_losses.append(np.mean(_train_losses))
            train_scores.append(np.mean(_train_scores))
            test_losses.append(np.mean(_test_losses))
            test_scores.append(np.mean(_test_scores))
            
    return train_losses, train_scores, test_losses, test_scores

def plot_c(cs, tr_loss, tr_score, te_loss, te_score, file, sweep_no):
    name = re.sub(r'\.csv', '', file)
    
    if sweep_no == 1:
        name = "c1_" + name
        xscale = "log"
    else:
        name = "c2_" + name
        xscale = "linear"
    
    # Plot 1
    plt.close()
    
    plt.axes(frameon=False)
    plt.grid(True)
    plt.xscale(xscale)

    plt.plot(cs, tr_score, label="Training accuracy")
    plt.plot(cs, te_score, label="Testing accuracy")

    plt.title("Regression accuracy per value of C")

    plt.xlabel("C")
    plt.ylabel("Accuracy")

    plt.legend()
    
    path = "images/logistic_acc_cv_"+name+".pdf"

    plt.savefig(path, format='pdf', dpi=300)
    
    # Plot 2
    plt.close()
    
    plt.axes(frameon=False)
    plt.grid(True, "both")
    plt.yscale("log")
    plt.xscale(xscale)

    plt.plot(cs, tr_loss, label="Training loss")
    plt.plot(cs, te_loss, label="Testing loss")

    plt.title("Regression loss per value of C")

    plt.xlabel("C")
    plt.ylabel("Logistic regression loss")

    plt.legend()
    
    path = "images/logistic_loss_cv_"+name+".pdf"

    plt.savefig(path, format='pdf', dpi=300)
    
def plot_dim(dims, train_losses, train_scores, test_losses, test_scores, file):
    name = re.sub(r'\.csv', '', file)
    
    # Plot 1
    plt.close()
    
    plt.axes(frameon=False)
    plt.grid(True)

    plt.plot(dims, np.array(train_scores), label="Training accuracy")
    plt.plot(dims, np.array(test_scores), label="Testing accuracy")

    plt.title("Regression accuracy per number of principal components")

    plt.xlabel("Number of principal components")
    plt.ylabel("Accuracy")

    plt.legend()
    
    path = "images/logistic_acc_cv_dim_"+name+".pdf"

    plt.savefig(path, format='pdf', dpi=300)
    
    # Plot 2
    plt.close()
    
    plt.axes(frameon=False)
    plt.grid(True, "both")
    plt.yscale("log")

    plt.plot(dims, train_losses, label="Training loss")
    plt.plot(dims, test_losses, label="Testing loss")

    plt.title("Regression loss per number of principal components")

    plt.xlabel("Number of principal components")
    plt.ylabel("Logistic regression loss")
    
    plt.tight_layout()

    plt.legend()
    
    path = "images/logistic_loss_cv_dim_"+name+".pdf"

    plt.savefig(path, format='pdf', dpi=300)

##################### Main #####################

files = [
    "train_c_red.csv",
    "train_c_red_bigram.csv",
    "train_c_red_stem.csv",
    "train_c_red_bigram_stem.csv"
]

path = "data/fake_news/"

if not os.path.exists("images"):
    os.makedirs("images")

for file in files:
    print("Doing " + file)
    
    # Load data and labels
    data = pd.read_csv(path+file, header=None)
    
    labels = data.iloc[:, -1]
    labels = labels.to_numpy()

    data = data.iloc[:, 1:-1]
    data = data.to_numpy()
    
    save_path = path + re.sub(r'\.csv', "", file)
    
    # First, coarse pass over C
    print("First pass over C")
    cs = np.float_power(10, np.arange(-4, 5))
    
    tr_loss, tr_score, te_loss, te_score = logistic_cv(data,
                                                       labels,
                                                       dims=[502], 
                                                       cs=cs)
    
    np.savez(save_path+"_c1.npz", 
             tr_loss=tr_loss, 
             tr_score=tr_score, 
             te_loss=te_loss, 
             te_score=te_score)
    plot_c(cs, tr_loss, tr_score, te_loss, te_score, file, sweep_no=1)
    
    # Second, fine-grained pass over C
    print("Second pass over C")
    cs2 = list(range(1, 101))
    
    tr_loss2, tr_score2, te_loss2, te_score2 = logistic_cv(data,
                                                           labels,
                                                           dims=[502], 
                                                           cs=cs2)
    
    np.savez(save_path+"_c2.npz", 
             tr_loss2=tr_loss2, 
             tr_score2=tr_score2, 
             te_loss2=te_loss2, 
             te_score2=te_score2)
    plot_c(cs2, tr_loss2, tr_score2, te_loss2, te_score2, file, sweep_no=2)
    
    # Pass over number of principal components
    print("Pass over number of principal components")
    dims = np.arange(0, 501, 50)

    train_losses, train_scores, test_losses, test_scores = logistic_cv(data,
                                                                       labels,
                                                                       dims=dims+2, 
                                                                       cs=[cs2[np.argmax(te_score2)]])
    
    np.savez(save_path+"_dims.npz", 
             train_losses=train_losses, 
             train_scores=train_scores, 
             test_losses=test_losses, 
             test_scores=test_scores)
    plot_dim(dims, train_losses, train_scores, test_losses, test_scores, file)
    
    