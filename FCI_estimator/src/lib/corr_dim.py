import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class TwoNN:
    """
    TwoNN estimator for intrinsic dimension.
    Here used as a term of comparison for the FCI estimator.
    """

    def __init__(self):
        self.d = None
        self.r = None

    def compute_dimension(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        N = X.shape[0]

        # compute nearest neighbors
        ngbrs = NearestNeighbors(n_neighbors=4, algorithm='auto')
        ngbrs.fit(X)

        # compute distance ratios
        ngbrs_distances, _ = ngbrs.kneighbors(X)
        indexes = list(range(N))
        for i in range(N):
            if ngbrs_distances[i, 1] == 0:
                indexes.remove(i)
        mu = ngbrs_distances[indexes, 2] / ngbrs_distances[indexes, 1]

        # estimate d (through max-likelihood)
        self.d = mu.shape[0] / np.sum(np.log(mu))

        # average distance from nearest neighbor
        self.r = np.mean(ngbrs_distances[indexes, 1])

        return self.d, self.r
