import numpy as np

from funcs import textToCleanedWords

class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def getAcc(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fn + self.fp)


class WordStat:

    # the None bit of the constructor is to allow for both initialization with, and
    # without, provided values
    # we only allow for initialization with both values, else we set it to 0
    #
    def __init__(self, reliable = None, unreliable = None):

        if (reliable == None or unreliable == None):
            self.reliable = 0
            self.unreliable = 0
        else:
            self.reliable = reliable
            self.unreliable = unreliable

def crossValidate(df, ngram, folds):

    performance = [0] * folds

    # As we want each fold to have an equal size
    dataUsed = len(df) - (len(df) % folds)

    print(f"Excluded {len(df) % folds} datapoints to ensure folds of equal size")


    rowsInFold =  int(dataUsed / folds)

    for i in range(0, folds):

        print(f"Work started on fold: {i + 1} out of {folds}")

        validationSet = df.iloc[i*rowsInFold : (i+1) * rowsInFold]
        validationSet = validationSet.reset_index()

        trainingSet = df.iloc[np.r_[:(i*rowsInFold), (i+1)*rowsInFold:]]
        trainingSet = trainingSet.reset_index()


        print("Calculating probabilities on training data")
        [wordCounts, numReliable, numUnreliable] = constructBayesCount(trainingSet, ngram)
        probs = wordCountsToProbabilities(wordCounts, numReliable, numUnreliable)

        print("Evaluating performance on validation set")
        cm = rateBayesPerformance(validationSet, probs, ngram)
        performance[i] = cm.getAcc()

        print(f"Finished with fold: {i  +1}, with an accuracy of: {performance[i]}")

    return performance




# Each entry in the wordCounts dict is the word, followed by a wordstat
# class keeping track of the counts, here
def addToBayesCount(wordCounts, words, unreliable, ngram):

    for i in range(len(words) - ngram + 1):
        # create the ngram key:
        token = '-'.join(words[i:i+ngram])

        if token not in wordCounts.keys():
            wordCounts[token] = WordStat()

        if (unreliable):
            wordCounts[token].unreliable += 1
        else:
            wordCounts[token].reliable += 1


def constructBayesCount(cleanedDataFrame, ngram=1):
    wordCounts = {}
    numReliable = 0
    numUnreliable = 0

    for i in range(len(cleanedDataFrame)):
        words = textToCleanedWords(cleanedDataFrame["text"][i])
        unreliable = cleanedDataFrame["label"][i]

        addToBayesCount(wordCounts, words, unreliable, ngram)

        # To keep track of the total number of words in each category
        if unreliable:
            numUnreliable += len(words) - ngram + 1
        else:
            numReliable += len(words) - ngram + 1


    return [wordCounts, numReliable, numUnreliable]

def wordCountsToProbabilities(wordCounts, numReliable, numUnreliable):

    wordProbs = {}
    tuningParam = 1.0 # Which helps us deal with cases in which we have little info on a word

    for word in wordCounts.keys():

        # Log probabilities are used as they have the useful probability that we can sum
        # instead of multiply to get the probability of multiple words, which is beneficial
        # if one or more of the words involved have a probability that is rather small

        logProbReliable = np.log((wordCounts[word].reliable + tuningParam) / numReliable);
        logProbUnReliable = np.log((wordCounts[word].unreliable + tuningParam) / numUnreliable)

        wordProbs[word] = WordStat(logProbReliable, logProbUnReliable)


    return wordProbs

def classifyAllBayesian(dataframe, wordProbs, ngram=1):
    classifications = []

    for text in dataframe["text"]:
        # We check if the input is valid and has enough tokens so that we
        # can reasonably infer something from the text itself
        if (not isinstance(text, str)):
            classifications.append(1)
            continue

        cleanedTokens = textToCleanedWords(text)

        if (len(cleanedTokens) < 10):
            classifications.append(1)
            continue

        classifications.append(classifyBayesian(text, wordProbs, ngram))

    return classifications


# Note that we assume here that the apriori probabilities are both 0.5,
# which reduces the amount of calculations we need to perform
# returns 1 if the text is unreliable (as this is the case for the labels)
# and 0 otherwise
def classifyBayesian(text, wordProbs, ngram):
    cleanedWords = textToCleanedWords(text)

    # Next we calculate the ratio P(reliable | text) / P(unreliable | text),
    # which gives us a ratio that tells us the classification
    #
    #As we use log probabilities, we sum instead of multiply

    totalReliable = 0
    totalUnreliable = 0


    for i in range(len(cleanedWords) - ngram + 1):
        token = '-'.join(cleanedWords[i:i+ngram])

        # Words that we have no information on are ignored
        if token in wordProbs.keys():
                probs = wordProbs[token]
                totalReliable += probs.reliable
                totalUnreliable += probs.unreliable

    if (totalUnreliable == 0):
        print(f"No unreliable tokens found, reliable: {totalReliable}")
        return 0

    if (totalReliable / totalUnreliable < 1.0):
        return 0

    return 1

def rateBayesPerformance(dataframe, wordProbs, ngram=1):

    # Here we define: Fake news = positive
    # like a diagnosis
    cm = ConfusionMatrix()

    for i in range(len(dataframe)):
        classification = classifyBayesian(dataframe["text"][i], wordProbs, ngram)

        if (classification == dataframe["label"][i]):
            if (classification == 1):
                cm.tp += 1
            else:
                cm.tn += 1
        else:
            if (classification == 1):
                cm.fp += 1
            else:
                cm.fn += 1

    return cm
