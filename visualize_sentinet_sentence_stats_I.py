import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from naive_bayesian.encoding import scoreTextInfoAll, scoreSentiAll, loadSenti,filterEnglish

"""
Exports Sentence statistics 1 and sentinet statistics for visualization.
"""

if __name__ == "__main__":
    train = "data/fake_news/train.csv"

    data = pd.read_csv(train,delimiter=",",header=0)
    data = filterEnglish(data)
    data.reset_index(inplace=True,drop=True)
    senti = loadSenti("naive_bayesian/sentiwordnet/SentiWordNet_3.0.0.txt")
    
    df_sentence = scoreTextInfoAll(data)
    df_senti = scoreSentiAll(data,senti)

    df_sentence.to_csv("data/fake_news/sentence_stats.csv",header=True,index=False)
    df_senti.to_csv("data/fake_news/senti_stats.csv",header=True,index=False)