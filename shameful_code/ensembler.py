import pandas as pd

#
# IN: A list of file locations, of which each is a csv with classifications for the
# training data. The length should be odd to ensure correctness (=no tie-breaks needed)
# Furthermore, the number of rows in each should be the same

def classifyEnsemble(files):

    if (len(files) % 2 == 0):
        print("Warning! Even number of files specified. The program may not function correctly")

    df = []

    for file in files:
        df.append(pd.read_csv(file))

    rows = len(df[0])

    for i in range(1, len(df)):
        if (len(df[i]) != rows):
            print("Error! Each file does not have the same amount of rows. Aborting")
            return

    classifications = []

    for i in range(0, rows):
        classification = 0
        for j in range(len(df)):
            classification += df[j]["label"][i]

        if (classification > len(df)/2):
            classifications.append(1)
        else:
            classifications.append(0)

    df2 = pd.DataFrame(zip(list(df[0]["id"]), classifications), columns=["id", "label"])
    df2.to_csv("ensemble_classification.csv", index=False)
