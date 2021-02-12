import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import statistics as stat
from langdetect import detect



# Scores all text in the dataframe using various metrics, (word length, sentence length, special character usage)
# and returns it in a new dataframe
#
# Note that all statistics are normalized using the maximum value found for each statistic
def scoreTextInfoAll(data):

    cols = ["senLengthMean", "senLengthStddev", "wordLengthMean", "wordLengthStddev", "# \"", "# \'", "# .", "# ,", "# !", "# ?", "# :", "label"]


    df = pd.DataFrame(columns = cols)

    resMax = [0] * len(cols);

    # We set the last value to 1 as it is the 'divisor' in normalisation for the label
    resMax[len(cols) - 1] = 1

    for i in range(0, len(data)):
        text = data["text"][i]
        label = data["label"][i]

        res = scoreTextInfo(text)

        # Store the largest values for each field found thusfar, for normalisation
        # length - 1, as resMax has 1 more entry, the 'normalisation' term for the label
        for j in range(0, len(resMax) - 1):
            if (res[j] > resMax[j]):
                resMax[j] = res[j]

        res = np.append(res, label)


        df = df.append(pd.DataFrame([res], columns = cols))

    # Normalisation step
    df = df / resMax

    #Turn the label back into an integer
    df["label"] = df["label"].astype(int)

    return df


# Given a single text, obtain some sentence statistics on it. Should not be used on its own, really
#
#Returns stats that are not normalized!
def scoreTextInfo(text):

    # First, we break it down to its sentences:
    sentences = sent_tokenize(text);
    sentenceLengths = []
    wordLengths = []

    specialChars = {"\"" : 0, "\'": 0, ".": 0, ",": 0, "!": 0, "?": 0, ":": 0}

    for sentence in sentences:
        uncleaned = word_tokenize(sentence)

        for word in uncleaned:
            if not word.isalpha():
                for char in word:
                    if not char.isalpha() and char in specialChars.keys():
                        specialChars[char] += 1;


        cleaned = textToCleanedWords(sentence)

        for word in cleaned:
            wordLengths.append(len(word))

            sentenceLengths.append(len(sentence))

    if (len(sentenceLengths) < 2) or (len(wordLengths) < 2):
        return np.array([stat.mean(sentenceLengths), sentenceLengths[0], stat.mean(wordLengths), wordLengths[0],
                        specialChars["\""], specialChars["\'"], specialChars["."], specialChars[","], specialChars["!"],
                        specialChars["?"], specialChars[":"]])
                        
    return np.array([stat.mean(sentenceLengths), stat.stdev(sentenceLengths), stat.mean(wordLengths), stat.stdev(wordLengths),
                        specialChars["\""], specialChars["\'"], specialChars["."], specialChars[","], specialChars["!"],
                        specialChars["?"], specialChars[":"]])

# Scores all text of the dataframe passed in using sentinet and returns
# a new dataframe with this new representation
def scoreSentiAll(data, senti):
    cols = ["# useful", "rel. useful", "posMean", "posStddev", "negMean", "negStddev", "objMean", "objStddev", "label"]

    df = pd.DataFrame(columns = cols)

    for i in range(0, len(data)):

        text = data["text"][i]
        cleanLemmas = cleanedWordsToLemmas(textToCleanedWords(text))

        label = data["label"][i]

        sentiScores = scoreSenti(cleanLemmas, senti)
        sentiScores = np.append(sentiScores, label)

        df = df.append(pd.DataFrame([sentiScores], columns = cols))

    return df


#Takes: A list of lemmas and an instantiated senti dictionary. Results in
# a bunch of derived statistics using sentinet
def scoreSenti(lemmas, senti):

    pos = []
    neg = []
    obj = []

    for lemma in lemmas:
        if lemma in senti.keys():
            cur = senti[lemma]
            pos.append(cur["pos"])
            neg.append(cur["neg"])
            obj.append(cur["obj"])

    if (len(pos) < 2):
        return np.array([len(pos), len(pos) / len(lemmas), 0, 0, 0, 0, 0, 0])

    return np.array([len(pos), len(pos) / len(lemmas), stat.mean(pos),
                        stat.stdev(pos), stat.mean(neg),
                        stat.stdev(neg), stat.mean(obj), stat.stdev(obj)])





# Takes in a dataframe and returns only the rows with English text
def filterEnglish(rawData):
    return rawData[findEnglish(rawData)]


# returns true at indices where the text is English, and false otherwise
# should not be directly accessed
def findEnglish(rawData):

    results = []

    totalWork = len(rawData["text"])
    workDone = 0

    fiveP = int(totalWork / 20)

    for row in rawData["text"]:

        if workDone % fiveP == 0:
            print(f"{int((workDone/fiveP) * 5)}%")

        # First, we check that the input is indeed a valid string: There are some invalid values
        # within the dataset

        if not isinstance(row, str):
            results.append(False)
            continue

        # We have some more invalid data, empty strings
        if (len(row.strip()) == 0):
            results.append(False)
            continue

        try:
            if detect(row) == 'en':
                results.append(True)
            else:
                results.append(False)
        #An exception is raised if it could not find tokens to extract at all in any language,
        #this is often the case with malformed data.
        except:
            results.append(False)

        workDone = workDone + 1

    return results

# Process as from: https://machinelearningmastery.com/clean-text-machine-learning-python/
# Takes in a string representing a text, and cleans the words by removing punctuation,
# other special symbols, and removing common words
def textToCleanedWords(text):

    wordTokens = word_tokenize(text)
    tokens = [w.lower() for w in wordTokens]

    #Create a mapping to filter out tokens that are only punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # Remove remaining tokens that contain non-letters
    words = [word for word in stripped if word.isalpha()]

    # Filter out common words that do not contribute much to meaning
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    return words

# Turns a list of cleaned words to their lemmatized form
def cleanedWordsToLemmas(words):
    # Own addition: Get the lemas:

    lemmatizer = WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(w) for w in words]

    return lemmas


# Loads in sentiwordnet from the provided text file

def loadSenti(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    return linesToDict(lines[:-1])

'''
    What follows are 3 helper functions to turn the text document into a dict that
    can be used for sentinet scoring

'''
def findNth(line, char, n):

    cur = 0

    for i in range(0, len(line)):
        c = line[i]

        if c == char:
            cur = cur + 1
            if (cur == n):
                return i

    return -1


def linesToDict(lines):
    res = {}

    oneP = int(len(lines) / 100)

    current = 0



    for line in lines:
        if (current % oneP == 0):
            print(f"{int(current/oneP)}%")

        current += 1

        if (line[0] == "#"):
            continue

        res.update(lineToEntry(line))


    return res



def lineToEntry(line):

    res = {}


    posScore = float(line[findNth(line, "\t", 2) + 1: findNth(line, "\t", 3)])
    negScore = float(line[findNth(line, "\t", 3) + 1: findNth(line, "\t", 4)])
    objScore = 1 - (posScore + negScore)

    words = []
    wordLine = line[findNth(line, "\t", 4) + 1: findNth(line, "\t", 5)]

    #First case is special:
    if (wordLine[:wordLine.find("#")].isalpha()):
        words.append(wordLine[:wordLine.find("#")])

    for i in range(1, wordLine.count("#")):
        initPos = findNth(wordLine, "#", i)

        #Skip leading number
        while wordLine[initPos] != " ":
            initPos = initPos + 1
        initPos += 1

        if (wordLine[initPos:findNth(wordLine, "#", i + 1)].isalpha()):
            words.append(wordLine[initPos:findNth(wordLine, "#", i + 1)])

    for word in words:
        res[word] = {"pos" : posScore, "neg" : negScore, "obj" : objScore}

    return res



'''
    ANEW didn't turn out to be too useful, but here it is for archive purposes


def prepareANEW(anewPath):
    anew = pd.read_csv(anewPath)

    anew["Description"] = anew["Description"].str.lower()

    return anew


def scoreANEW(lemmas, anew):
    #ANEW provides us with:
    #
    #   Valence:    Happy-unhappy scale
    #   Arousal:    Stimulated - calm scale
    #   Dominance:  Controlling - loose scale
    #
    #   Rating scales are 9-point, hence a division by 9 for normalisation

    val = []
    ar = []
    dom = []

    for lemma in lemmas:

        row = anew[anew["Description"] == lemma]

        #If we have no information on a lemma, we skip
        if row.size == 0:
            continue

        val.append(row["Valence Mean"].values[0])
        ar.append(row["Arousal Mean"].values[0])
        dom.append(row["Dominance Mean"].values[0])


    #We get the mean and stddev of each known word score, as well as how many words
    #were found that could be used for information from ANEW
    if (len(val) < 2):
        print("Warning: Not enough information contained to assign any score")
        return np.array([len(val), len(val) / len(lemmas), 0, 0, 0, 0, 0, 0])

    return np.array([len(val), len(val) / len(lemmas),
                        stat.mean(val), stat.stdev(val),
                        stat.mean(ar), stat.stdev(ar),
                        stat.mean(dom), stat.stdev(dom)
                    ])
'''
