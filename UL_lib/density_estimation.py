import numpy as np
import pandas as pd


# histogram estimator using Freedman-Diaconis rule
class HistEstimator:

    def __init__(self):
        self.h = None
        self.n_bins = None
        self.xmins = None
        self.pdf = None

    def fit(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        if len(X.shape) == 1:  # reshape single array
            X = X.reshape(1, -1)
        N = X.shape[0]
        n_features = X.shape[1]

        # compute bins (using Freedman-Diaconis rule)
        self.h = np.empty(n_features)  # bin width
        for i in range(n_features):
            q75, q25 = np.percentile(X[:, i], [75, 25])
            self.h[i] = 2 * (q75 - q25) / (N ** (1/3))
        self.n_bins = np.empty(n_features, dtype=int)  # number of bins
        for i in range(n_features):
            self.n_bins[i] = int(round((X[:, i].max() - X[:, i].min()) / self.h[i]))

        # save minimum values of each feature
        self.xmins = np.empty(n_features)
        for i in range(n_features):
            self.xmins[i] = X[:, i].min()

        # estimate PDF (it has the form of a list of arrays each one
        # corresponding to a feature)
        self.pdf = []
        for i in range(n_features):
            j = ((X[:, i]-self.xmins[i]) / self.h[i]).astype(int)
            j[j==self.n_bins[i]] = self.n_bins[i]-1  # adjust last bin
            hist = np.zeros(self.n_bins[i])
            for idx in j:
                hist[idx] += 1
            self.pdf.append(hist)

    def return_pdf(self):
        return self.pdf, self.n_bins

    def compute_proba(self, x):
        # compute indexes of bins for each feature
        j = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            j[i] = int((x[i]-self.xmins[i]) / self.h[i])

        # compute probability
        proba = 1
        for i in range(x.shape[0]):
            proba *= self.pdf[i][j[i]]

        return proba


# kernel density estimator using Gaussian kernel
class GaussKDEstimator:

    def __init__(self):
        self.X = None
        self.N = None
        self.h = None

    def fit(self, data):
        self.X = None
        if isinstance(data, pd.DataFrame):
            self.X = data.values
        else:
            self.X = data.copy()
        if len(self.X.shape) == 1:  # reshape single array
            self.X = self.X.reshape(1, -1)
        self.N = self.X.shape[0]
        n_features = self.X.shape[1]

        # compute bins (using Silverman's rule)
        self.h = np.empty(n_features)  # bin width
        for i in range(n_features):
            q75, q25 = np.percentile(self.X[:, i], [75, 25])
            self.h[i] = 0.9 * min(np.std(self.X[:, i]), (q75-q25)/1.34) * self.N ** (-1/5)

    def return_h(self):
        return self.h

    def compute_proba(self, x):
        # compute probability (using Gaussian kernel)
        proba = 0
        for i in range(self.N):
            proba += np.exp(-0.5 * np.sum(((x - self.X[i]) / self.h) ** 2))
        proba /= (self.N * np.prod(self.h) * np.sqrt(2 * np.pi))

        return proba
