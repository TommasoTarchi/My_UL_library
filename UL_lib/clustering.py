import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


class KMeans:
    """
    This is an implementation of the k-means clustering algorithm.
    Initialization of the clusters can be performed using both "random"
    and "k-means++" procedures.
    "Medoids" can be used as well.

    Inputs:
    - n_clusters = number of clusters
    - init = cluster initiaization method ("random" or "k-means++")
    - n_init = number of times the clustering is run
    - max_iter = maximum number of iterations per run
    - medoids = whether to use or not "K-medoids"

    Methods can be used to return:
    - labels assigned to train data (return_labels)
    - loss of best clustering (return_loss)
    - labels assigned to new data (predict)
    """

    def __init__(self, n_clusters=8, init='random', n_init=1, max_iter=100, medoids=False):
        self.n_clusters = n_clusters
        self.init = init  # initialization method ('random' or 'k-means++')
        self.n_init = n_init  # number of initializations (i.e. times we run clustering)
        self.max_iter = max_iter  # maximum number of iterations for single run
        self.medoids = medoids  # perform K-medoids or not
        self.best_labels = None
        self.centroids = None
        self.min_loss = None

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

        # run k-means multiple times
        for _ in range(self.n_init):

            if self.init == 'random':
                # initialize centroids randomly
                init_centroids = np.random.choice(np.arange(N), size=self.n_clusters, replace=False)

            elif self.init == 'k-means++':
                # initialize centroids using k-means++
                init_centroids = np.empty(self.n_clusters, dtype=int)
                init_centroids[0] = np.random.choice(np.arange(N))
                for i in range(1, self.n_clusters):
                    mask = np.ones(X.shape[0], dtype=bool)
                    mask[init_centroids[:i]] = False
                    X_masked = X[mask]
                    distances = np.empty(N-i)
                    for j in range(N-i):
                        distances[j] = np.min(np.linalg.norm(X_masked[j] - X[init_centroids[:i]], axis=1))
                    probabilities = distances**2 / (distances**2).sum()
                    init_centroids[i] = np.random.choice(np.arange(N-i), p=probabilities)

            # compute distances to centroids
            distances = np.empty((N, self.n_clusters))
            for i in range(self.n_clusters):
                distances[:, i] = np.linalg.norm(X - X[init_centroids[i]], axis=1)

            # assign labels
            labels = np.argmin(distances, axis=1)

            # update centroids
            self.centroids = np.empty((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                if np.any(labels == i):
                    self.centroids[i] = np.mean(X[labels == i])
                else:
                    self.centroids[i] = X[init_centroids[i]]

            # use medoids if required (medoids are saved in place of centroids)
            if self.medoids:
                for i in range(self.n_clusters):
                    if np.any(labels == i):
                        X_cluster = X[labels == i].copy()
                        dist = np.linalg.norm(X_cluster-self.centroids[i], axis=1)
                        medoid_idx = np.argmin(dist)
                        self.centroids[i] = X_cluster[medoid_idx]

            # update centroids until convergence
            new_labels = np.empty(N)
            iter = 0
            while np.any(new_labels != labels):

                labels = new_labels.copy()

                for i in range(self.n_clusters):
                    distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
                new_labels = np.argmin(distances, axis=1)

                for i in range(self.n_clusters):
                    self.centroids[i] = np.mean(X[new_labels == i], axis=0)

                iter += 1

                if iter == self.max_iter:
                    break

            # compute loss (inertia, using squared distance from centroids)
            loss = 0
            for i in range(self.n_clusters):
                loss += np.sum(np.linalg.norm(X[labels == i] - self.centroids[i], axis=1)**2)

            # save best labels
            if self.best_labels is None or loss < self.min_loss:
                self.best_labels = labels.copy()
                self.min_loss = loss

    def return_labels(self):
        return self.best_labels

    def return_loss(self):
        return self.min_loss

    def predict(self, data):
        """
        This method can be used to assign new datapoint to computed
        clusters.
        """
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        if len(X.shape) == 1:  # reshape single array
            X = X.reshape(1, -1)
        N = X.shape[0]

        # compute distances to centroids
        distances = np.empty((N, self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)

        # assign labels
        return np.argmin(distances, axis=1)


class FuzzyCMeans:
    """
    This is an implementation of the fuzzy c-means clustering algorithm.

    Inputs:
    - n_clusters = number of clusters
    - f = fuzzification parameter
    - n_init = number of times the clustering is run
    - max_iter = maximum number of iterations per run
    - epsilon = threshold parameter for convergence

    Methods can be used to return:
    - memebership matrix for train data (return_U)
    - loss of best clustering (return_loss)
    - membership matrix for new data (predict)
    """

    def __init__(self, n_clusters=8, f=2, n_init=1, max_iter=100, epsilon=1e-4):
        self.n_clusters = n_clusters
        self.f = f  # fuzzification parameter
        self.n_init = n_init  # number of initializations (i.e. times we run clustering)
        self.max_iter = max_iter  # maximum number of iterations for single run
        self.epsilon = epsilon  # convergence threshold
        self.best_U = None
        self.centroids = None
        self.min_loss = None

    def fit(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        N = X.shape[0]
        n_features = X.shape[1]

        for _ in range(self.n_init):

            # choose degrees of membership randomly
            U = np.random.rand(N, self.n_clusters)
            for i in range(N):
                U[i, :] /= U[i, :].sum()

            iter = 0
            while True:

                # update centroids
                self.centroids = np.empty((self.n_clusters, n_features))
                for i in range(self.n_clusters):
                    self.centroids[i] = (U[:, i]**self.f).reshape(-1, 1).T @ X / (U[:, i]**self.f).sum()

                # compute distances to centroids
                distances = np.empty((N, self.n_clusters))
                for i in range(self.n_clusters):
                    distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)

                # update degrees of membership
                new_U = np.empty((N, self.n_clusters))
                for i in range(N):
                    for j in range(self.n_clusters):
                        exponent = 2 / (self.f-1)
                        if np.any(distances[i, j] != 0):
                            temp = self.n_clusters * (distances[i, j]**exponent)
                            temp /= np.sum(distances[i, :]**exponent)
                            new_U[i, j] = temp ** -1
                        else:
                            new_U[i, j] = 0
                for i in range(N):
                    new_U[i, :] /= new_U[i, :].sum()  # renormalization

                iter += 1

                if iter == self.max_iter or np.linalg.norm(U-new_U) < self.epsilon:
                    U = new_U.copy()
                    break

                U = new_U.copy()

            # compute loss
            loss = 0
            for i in range(self.n_clusters):
                loss += np.sum(U[:, i] * (np.linalg.norm(X - self.centroids[i], axis=1)**2))

            # save best degrees of membership
            if self.best_U is None or loss < self.min_loss:
                self.best_U = U.copy()
                self.min_loss = loss

    def return_U(self):
        return self.best_U

    def return_loss(self):
        return self.min_loss

    def predict(self, data):
        """
        This method can be used to assign new datapoint to computed
        clusters.
        """
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        if len(X.shape) == 1:  # reshape single array
            X = X.reshape(1, -1)
        N = X.shape[0]

        # compute distances to centroids
        distances = np.empty((N, self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)

        # compute degrees of membership
        U = np.empty((N, self.n_clusters))
        for i in range(N):
            for j in range(self.n_clusters):
                exponent = -2 / (self.f-1)
                U[i, j] = (distances[i, j] / np.sum(distances[i])) ** exponent
        for i in range(N):
            U[i, :] /= U[i, :].sum()  # renormalization

        return U


class SpectralClustering:
    """
    This is an implementation of the spectral clustering algorithm.
    The graph for performing clustering can be built using k-nearest neighbors,
    epsilon-ball or as simple fully-connected graph.
    k-means algorithm can be initialized at random or using k-means++.

    Inputs:
    - n_clusters = number of clusters
    - build_graph = method to build the graph (kNN, epsilon-ball or fully-connected graph)
    - k = paramter for kNN graph construction
    - epsilon = parameter for epsilon-ball graph construnction
    - sigma = parameter for similarity graph construction
    - kmeans_init = method fot initialization of the k-means algorithm (random or k-means++)
    - normalize = whether to normalize or not the Laplacian matrix
    """

    def __init__(self, n_clusters=8, build_graph='kNN', k=5, epsilon=0.5, sigma=1, kmeans_init='random', normalize=False):
        self.n_clusters = n_clusters
        self.build_graph = build_graph  # method to build graph ('kNN', 'eps-ball', 'fully-connected')
        self.k = k  # parameter for kNN graph construction
        self.epsilon = epsilon  # parameter for epsilon-ball graph construction
        self.gamma = 1 / sigma**2  # parameter for similarity graph construction
        self.kmeans_init = kmeans_init  # initialization method for k-means ('random', 'k-means++')
        self.normalize = normalize  # whether we use normalized or non normalized Laplacian
        self.labels = None

    def fit(self, data):
        X = None
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data.copy()
        if len(X.shape) == 1:  # reshape single array
            X = X.reshape(1, -1)
        N = X.shape[0]

        E = np.zeros((N, N))  # similarity graph

        if self.build_graph == 'kNN':
            # construct graph connecting kNNs
            ngbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto')
            ngbrs.fit(X)
            E = np.triu(ngbrs.kneighbors_graph(X, mode='distance').toarray())

            # build similarity graph
            for i in range(N):
                for j in range(i+1, N):
                    if E[i, j] != 0:
                        E[i, j] = np.exp(-self.gamma * E[i, j]**2)
            E = E + E.T
            for i in range(N):
                E[i, i] = 0

        elif self.build_graph == 'eps-ball':
            # compute distance matrix
            D = squareform(pdist(X, metric='euclidean'))

            # build graph
            for i in range(N):
                for j in range(i+1, N):
                    if D[i, j] < self.epsilon:
                        E[i, j] = np.exp(-self.gamma * D[i, j]**2)
            E = E + E.T

        elif self.build_graph == 'fully-connected':
            # compute distance matrix
            D = squareform(pdist(X, metric='euclidean'))

            # build graph
            for i in range(N):
                for j in range(i+1, N):
                    E[i, j] = np.exp(-self.gamma * D[i, j]**2)
            E = E + E.T

        # build Laplacian matrix
        D = np.diag(np.sum(E, axis=1))
        D_max = np.max(D)
        for i in range(N):
            if D[i, i] == 0:
                D[i, i] = 1e-9 * D_max  # to avoid division by zero
        L = D - E

        # use normalized Laplacian
        if self.normalize:
            D_sqrt_inv = np.diag(1 / np.sqrt(np.diag(D)))
            L = D_sqrt_inv @ L @ D_sqrt_inv

        # find eigenvectors of Laplacian
        evals, evecs = np.linalg.eig(L)
        evals_order = np.argsort(evals)
        evecs = evecs[:, evals_order][:, 1:self.n_clusters+1]

        # apply k-means to eigenvectors
        kmeans = None
        if self.kmeans_init == 'random':
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        elif self.kmeans_init == 'k-means++':
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, init='k-means++')
        kmeans.fit(evecs)
        self.labels = kmeans.return_labels()

    def return_labels(self):
        return self.labels


class DensPeakClustering:
    """
    This is an implementation of the density peak clustering algorithm,
    using a Gaussian distance kernel.

    Inputs:
    - dc = parameter for distance kernel

    Methods can be used to return:
    - scatterplot of rho VS delta for cluster centroids choice (scatterplot)
    - automatically assigned labels on the base of previous scatterplot (automatic_clustering)
    """

    def __init__(self, dc):
        self.dc = dc  # parameter for distance kernel
        self.X = None  # data
        self.sorted_indexes = None
        self.sorted_rho = None
        self.sorted_delta = None
        self.delta_indexes = None  # indexes of nearest neighbors of higher density

    def fit(self, data):
        self.X = None
        if isinstance(data, pd.DataFrame):
            self.X = data.values
        else:
            self.X = data.copy()
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(1, -1)
        N = self.X.shape[0]

        # compute distance matrix
        D = squareform(pdist(self.X, metric='euclidean'))

        # compute densities
        rho = np.zeros(N)
        for i in range(N):
            rho[i] = np.sum(np.exp(-(np.delete(D[i, :], i) / self.dc)**2))

        # sort datapoints by density
        self.sorted_indexes = np.argsort(rho)[::-1]

        # compute distances delta
        delta = np.zeros(N)
        self.delta_indexes = np.empty(N, dtype=int)
        max_delta = 0  # needed to compute delta of point with highest density
        for i in range(N):
            ordered_idx = np.where(self.sorted_indexes==i)[0][0]
            if ordered_idx != 0:  # if not point with highest density
                delta[i] = np.min(D[i, self.sorted_indexes[:ordered_idx]])
                delta_idx = np.argmin(D[i, self.sorted_indexes[:ordered_idx]])
                self.delta_indexes[i] = self.sorted_indexes[delta_idx]
            else:  # if point with highest density
                self.delta_indexes[i] = N  # arbitrary value (any larger than N-1 would work)
            if delta[i] > max_delta and ordered_idx != 0:
                max_delta = delta[i]
        delta[self.sorted_indexes[0]] = 1.05 * max_delta

        # generate arrays for scatterplot
        self.sorted_rho = rho[self.sorted_indexes]
        self.sorted_delta = delta[self.sorted_indexes]

    def scatterplot(self):
        """
        This method can be used to return the scatterplot of rho
        VS delta for cluster centroids choice.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.sorted_rho, self.sorted_delta, s=10) 
        plt.xlabel(r"$\rho$")
        plt.ylabel(r"$\delta$")
        plt.title(f"Scatterplot of rho vs delta with dc={self.dc}")
        plt.show()

    # select clusters automatically (threshold is a percentage
    # of the maximum value of delta)
    #
    # computes centroids and clusters input data accordingly
    def automatic_clustering(self, thres=0.25):
        """
        This method can be used to automatically cluster data without visual
        inspection of the rho VS delta scatterplot.
        Cluster centroids are chosen using a threshold on difference in delta
        between consecutive points.

        Inputs:
        thres = threshold for centroids choice (percentage of maximum delta)

        Outputs:
        labels = labels by clustering
        """
        # compute absolute threshold from percentage
        abs_thres = thres * np.max(self.sorted_delta)

        # compute differences in delta
        diff = np.diff(self.sorted_delta)

        # select centroids
        centroids = []
        centroids.append(self.sorted_indexes[0])
        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] < 0 and self.sorted_delta[i] > abs_thres:
                centroids.append(self.sorted_indexes[i])
        centroids = np.array(centroids)

        # build tree used to assign labels more efficiently
        index_tree = []  # tree structure to store indexes of datapoints ("reverse" of delta_indexes)
        for i in range(self.X.shape[0]):
            index_tree.append(np.where(self.delta_indexes==i)[0].tolist())

        # assign labels (using a "recursive" approach)
        labels = np.full(self.X.shape[0], -1, dtype=int)
        labels[centroids] = np.arange(centroids.shape[0])  # assign labels to centroids
        todo_indexes = centroids.tolist()  # list to store next indexes to label
        for cent in todo_indexes:  # remove centroids from tree (already labeled)
            if cent != self.sorted_indexes[0]:
                index_tree[self.delta_indexes[cent]].remove(cent)
        while todo_indexes:
            todo_indexes_temp = []
            for i in todo_indexes:
                for j in index_tree[i]:
                    labels[j] = labels[i]
                todo_indexes_temp += index_tree[i]
            todo_indexes = todo_indexes_temp

        return labels
