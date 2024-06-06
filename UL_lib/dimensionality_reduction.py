import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors


# PCA
class PCA:

    def __init__(self):
        self.eval = None
        self.evec = None
        self.sing_values = None
        self.exp_var = None

    def fit(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        N = X.shape[0]

        # standardize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # compute covariance matrix
        C = X.T @ X
        C /= N

        # find eigenvalues, eigenvectors and singular values of covariance matrix
        self.eval, self.evec = np.linalg.eig(C)
        self.sing_values = np.sqrt(self.eval)

        # rearrange outputs in descending order
        PC_order = np.argsort(self.eval)[::-1]
        self.eval = self.eval[PC_order]
        self.evec = self.evec[:, PC_order]
        self.sing_values = self.sing_values[PC_order] * np.sqrt(N)

        # compute explained variance ratio
        self.exp_var = self.eval / self.eval.sum()

    def return_PCs(self):
        return self.eval, self.evec

    def return_sing_values(self):
        return self.sing_values

    def return_explained_var(self):
        return self.exp_var

    def project(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()

        # standardize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # project data along PCs
        Y = X @ self.evec

        return Y


# isomap
#
# we use kNNs to construct the grah, euclidean distance as metric,
# and Floyd-Warshall algorithm to compute distances on manifold
class Isomap:

    def __init__(self, n_components=2, k=5):
        self.n_components = n_components
        self.k = k  # number of nearest neighbours
        self.eval = None
        self.evec = None

    def fit(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        N = X.shape[0]

        # construct graph connecting kNNs
        ngbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto')
        ngbrs.fit(X)
        E = np.triu(ngbrs.kneighbors_graph(X, mode='distance').toarray())
        E = E + E.T
        max_dist = np.max(E)  # needed for normalization
        E[E == 0.] = np.inf
        for i in range(N):
            E[i, i] = 0.

        # normalize distances
        E = E / max_dist

        # drop datapoints without connections
        inf_rcs = np.sum(np.isinf(E), axis=1)
        valid_rcs = np.where(inf_rcs < E.shape[1] - 1)[0]
        E = E[valid_rcs, :][:, valid_rcs]
        new_N = E.shape[0]  # number of points after dropping ones without connections
        print(f"{N-new_N} datapoints had no connection and were dropped")

        # apply Floyd-Warshall
        for k in range(new_N):
            for i in range(new_N):
                for j in range(new_N):
                    if E[i, j] > E[i, k] + E[k, j]:
                        E[i, j] = E[i, k] + E[k, j]

        #new_N = N  # number of points after dropping ones with infinite distances

        # check for infinite distances and drop corresponding datapoints
        #if np.any(np.isinf(E)):
        #    i = 0
        #    while i<N:
        #        try:
        #            if np.any(np.isinf(a[i, :])):
        #                a = np.delete(a, i, axis=0)
        #                a = np.delete(a, i, axis=1)
        #                new_N -= 1
        #            else:
        #                i += 1
        #        except:
        #            break
        #
        #    print("""ATTENTION: the given value of k was not large enough to
        #            compute distance between all pairs of datapoints\nplease
        #            use a larger value ->""")
        #    print(f"\t-> {N-new_N} datapoints were dropped")

        print(f"number of nans in E: {np.sum(np.isnan(E))}")
        print(f"number of infs in E: {np.sum(np.isinf(E))}")

        # compute Gram matrix (double-center distance matrix)
        E = E**2
        E_sum = E.sum()
        E_sums_rows = np.empty(new_N)
        for i in range(new_N):
            E_sums_rows[i] = E[i, :].sum()
        G = np.empty((new_N, new_N))
        for i in range(new_N):
            for j in range(i):
                if (E_sums_rows[i] + E_sums_rows[j] - E_sum) == 0:
                    G[i, j] = 0
                else:
                    G[i, j] = (-E[i, j] + (E_sums_rows[i] + E_sums_rows[j] - E_sum) / new_N) / 2
            G[i, i] = 0
        G = G + G.T #- np.diag(np.diag(G))

        #G = -0.5 * (E ** 2 - (1 / new_N) * np.outer(np.ones(new_N), np.diag(E)) -
        #            (1 / new_N) * np.outer(np.diag(E), np.ones(new_N)) +
        #            (1 / (new_N ** 2)) * np.outer(np.ones(new_N), np.ones(new_N)) * np.sum(E))

        print(f"number of negative values in G: {np.sum(G<0)}")
        print(f"number of nans in G: {np.sum(np.isnan(G))}")
        print(f"number of infs in G: {np.sum(np.isinf(G))}")

        # find eigenvalues and eigenvectors of Gram matrix
        self.eval, self.evec = np.linalg.eig(G)

        # select desired components
        comp_order = np.argsort(self.eval)[::-1]
        self.eval = self.eval[comp_order]
        self.eval = self.eval[:self.n_components]
        self.evec = self.evec[:, comp_order]
        self.evec = self.evec[:, :self.n_components]

    def project(self):
        self.eval = np.sqrt(self.eval)
        sqrt_eval = np.diag(self.eval)

        return self.evec @ sqrt_eval


# kernel-PCA
class KernelPCA:

    def __init__(self, n_components='all', kernel='linear', gamma=1, degree=3, coef0=0):
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = 1 / np.sqrt(2*gamma)
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.eval = None
        self.evec = None

    def fit_project(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        N = X.shape[0]

        # standardize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # compute kernel matrix
        K = np.zeros((N, N))
        if self.kernel == 'linear':
            K = X @ X.T
        elif self.kernel == 'rbf':
            for i in range(N):
                for j in range(i+1, N):
                    K[i, j] = np.exp(-minkowski(X[i], X[j])**2 / (2*self.sigma**2))
            K = K + K.T + np.eye(N)
        elif self.kernel == 'poly':
            K = (self.gamma * X @ X.T + self.coef0) ** self.degree

        # compute Gram matrix (double-center kernel matrix)
        ones_mat = np.ones((N, N)) / N
        G = K - ones_mat @ K - K @ ones_mat + ones_mat @ K @ ones_mat

        # find eigenvalues and eigenvectors of Gram matrix
        self.eval, self.evec = np.linalg.eig(G)
        self.eval = np.real(self.eval)  # to avoid imaginary parts due to numerical errors
        self.evec = np.real(self.evec)  # to avoid imaginary parts due to numerical errors

        # select desired components
        comp_order = np.argsort(self.eval)[::-1]
        self.eval = self.eval[comp_order]
        self.evec = self.evec[:, comp_order]
        if self.n_components == 'all':
            self.n_components = self.eval.shape[0]
        self.eval = self.eval[:self.n_components]
        self.evec = self.evec[:, :self.n_components]

        # project data along PCs
        self.eval = np.sqrt(self.eval)
        sqrt_eval = np.diag(self.eval)

        return self.evec @ sqrt_eval


# two-NN estimator for intrinsic dimensionality
class TwoNN:

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
