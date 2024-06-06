import numpy as np
import pandas as pd


# function to compute mutual information between two CATEGORICAL variables
#
# takes to numpy arrays of integers as input
def compute_MI(x, y):
    N = x.shape[0]

    # compute empirical distributions
    px = np.bincount(x) / N
    py = np.bincount(y) / N
    pxy = np.zeros((px.shape[0], py.shape[0]))
    for i in range(N):
        pxy[x[i], y[i]] += 1
    pxy /= N

    # compute mutual information
    mi = 0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] != 0 and px[i] != 0 and py[j] != 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
            else:
                mi += 0

    return mi


# function to compute normalized mutual information
#
# takes to numpy arrays of integers as input representing the two
# sets (GT and computed by clustering) of labels
def compute_NMI(gt, labels):
    N = gt.shape[0]

    # compute mutual information
    mi = compute_MI(gt, labels)

    # compute empirical distributions
    p_gt = np.bincount(gt) / N
    p_labels = np.bincount(labels) / N

    # compute entropies
    h_gt = 0
    for i in range(p_gt.shape[0]):
        if p_gt[i] != 0:
            h_gt -= p_gt[i] * np.log(p_gt[i])
    h_labels = 0
    for i in range(p_labels.shape[0]):
        if p_labels[i] != 0:
            h_labels -= p_labels[i] * np.log(p_labels[i])

    # compute normalized mutual information (MI divided by the mean of the two entropies)
    nmi = 2 * mi / (h_gt + h_labels)

    return nmi


# function to compute F-ratio
#
# takes as input a numpy matrix (or a pandas dataframe) representing
# the datapoints and an array of integers representing the labels
def compute_FRatio(data, labels):
    X = None
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data.copy()
    if len(X.shape) == 1:  # reshape single array
        X = X.reshape(1, -1)
    n_clusters = np.unique(labels).shape[0]

    # compute centroids
    centroids = np.empty((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centroids[i] = np.mean(X[labels == i], axis=0)

    # compute intra-cluster variance
    SSW = 0
    for i in range(n_clusters):
        SSW += np.sum(np.linalg.norm(X[labels == i] - centroids[i], axis=1) ** 2)

    # compute inter-cluster variance
    SSB = 0
    X_mean = np.mean(X, axis=0)
    for i in range(n_clusters):
        SSB += np.sum(labels==i) * np.linalg.norm(centroids[i] - X_mean) ** 2

    # compute F-ratio
    F = n_clusters * SSW / SSB

    return F
