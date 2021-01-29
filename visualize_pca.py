from scratch_builds.pca import PCA
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Read in data
    data = pd.read_csv("data/vectorized.csv",delimiter=",",header=None)
    data = data.to_numpy()
    data = data[:data.shape[0]-1000]
    test = data[data.shape[0]-1000:]

    # Get data into right format and perform pca
    data = data.T
    test = test.T
    pca = PCA(1)
    pca.fit(data)

    pca.update_complexity(recalculate=True)
    msq, diss_r = pca.calculate_stats(pca.m)

    validation_loss = []
    msqs = []
    diss = []
    # Calculate relative dissimilarity for up to 2000 components retained
    for i in range(1999):
        pca.update_complexity(recalculate=True)
        msq, diss_r = pca.calculate_stats(pca.m)
        loss = pca.calculate_loss(test)
        validation_loss.append(loss)
        msqs.append(msq)
        diss.append(diss_r)
    
    
    plt.plot(range(len(validation_loss)),validation_loss,color="blue")
    plt.title("Quadratic loss between patterns and reconstructions from validation set")
    plt.xlabel("Number of principal components retained")
    plt.ylabel("Quadratic loss")
    plt.show()
    
    plt.plot(range(len(msqs)),msqs,color="red")
    plt.title("Mean square between patterns and reconstructions")
    plt.xlabel("Number of principal components retained")
    plt.ylabel("Mean-square distance")
    plt.show()

    plt.plot(range(len(diss)),diss,color="green")
    plt.title("Relative amount of dissimilarity")
    plt.xlabel("Number of principal components retained")
    plt.ylabel("Relative Dissimilarity")
    plt.show()
    
    