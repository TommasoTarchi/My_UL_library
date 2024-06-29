import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.constants import pi
from scipy.special import gamma, hyp2f1
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import minkowski
from scipy.optimize import curve_fit
import mpmath


# used in FCIEstimator
def preprocess(data: np.ndarray):
    """
    This function centers and normalize a dataset.
    The dataset is expected to be passed in the form of a numpy array
    with one row per datapoint and one column per feature.
    NOTICE that 2D images data are flattened.
    """

    if data.ndim != 2:
        data = data.reshape(data.shape[0], -1)

    # center data
    mass_center = np.mean(data, axis=0, dtype=data.dtype)
    data_centered = data - mass_center

    # normalize data
    norms = np.linalg.norm(data_centered, axis=1)
    norms[norms == 0.0] = 1.0  # possible null datapoint
    data_normalized = data_centered / norms.reshape(-1, 1)

    return data_normalized

# used in FCIEstimator
def approximate_angle_ratio(dim: float):
    """
    This function can be used to compute the approximated angle
    ratio in estimate_CI (following).
    The formula is based on Stirling approximation applied to both
    numerator and denominator.
    As all approximations, only works for large angles (above ~300).
    """

    if dim < 339:
        print("Warning: using Stirling approximation for angle ratio, but dimension small enough to use exact computation of angles")

    ratio = np.exp((0.5*dim - 0.5) * np.log(dim - 2.) - (0.5*dim - 1.) * np.log(dim - 3.))
    angle_ratio = 0.5 * ratio / np.sqrt(2. * np.pi * np.exp(1.))

    return angle_ratio

# used in FCIEstimator
def compute_empirical_FCI(data: np.ndarray, r: np.ndarray):
    """
    This function computes the density of neighbours (also known as
    "empirical correlation integral" in the literature) of a dataset,
    for a given list of cutoff distances (passed in the form of a numpy
    array r).
    Data are expected to be passed in the form of a numpy array with
    one row per datapoint and one column per feature.
    """

    dset_size = data.shape[0]

    # compute normalization factor
    norm = 1 / dset_size / (dset_size-1)

    # compute distances between all couples of datapoints
    distances = squareform(pdist(data))

    # compute number of neighbours within cutoff distances
    accumulate_distance = np.zeros_like(r, dtype=np.float64)
    for i, cutoff in enumerate(r):
        accumulate_distance[i] = (np.sum(distances <= cutoff) - dset_size)

    return accumulate_distance * norm

# used in FCIEstimator
def estimate_FCI(r: np.ndarray, d: float):
    """
    This function estimates the correlation integral for a given cutoff
    radius r, rescaled of r_s in dimension d.
    It is used in the non linear fit for the model.
    """

    # precompute ratio between solid angles (for dimensions larger
    # than 320 an approximation based on Stirling formula is used
    # for numerical stability (gamma function would give result 0.0))
    if d < 339:
        solid = 2. * pi**(0.5 * d) / gamma(0.5 * d)  # solid d-dimensional angle
        solid_1 = 2. * pi**(0.5 * (d-1)) / gamma(0.5 * (d-1))  # solid (d-1)-dimensional angle
        angle_ratio = 0.5 * solid_1 / solid
    else:
        angle_ratio = approximate_angle_ratio(d)

    # precompute and normalize last arg (cannot be larger than 1)
    last_arg = (r**2 - 2) ** 2
    last_arg /= np.max(last_arg)

    # compute (2,1)-hypergeometric function
    #
    # (for d > 340 we use mpmath method for hypergeometric,
    # since scipy does not work anymore; notice that mpmath
    # method is not vectorizable and requires typecasting)
    with np.errstate(divide='ignore', invalid='ignore'):
        if d < 339:
            hypergeom = hyp2f1(0.5, 1. - 0.5 * d, 1.5, last_arg)
        else:
            hypergeom = np.empty_like(last_arg, dtype=np.float64)
            for i in range(hypergeom.shape[0]):
                hypergeom[i] = float(abs(mpmath.hyp2f1(0.5, 1. - 0.5*d, 1.5, last_arg[i])))

    return 0.5 + angle_ratio * (r**2 - 2) * hypergeom


class PCA:
    """
    This is an implementation of principal component analysis.
    PCs to retain can be decided a posteriori by looking at the
    computed explained variance.

    Methods can be used to return:
    - principal components (return_PCs)
    - explained variance of data (return_explained_var)
    - projection of a new datapoint along PCs (project)
    """

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
    """
    WARNING: this method is still under revision, it is not guaranteed to work.
    """

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


class KernelPCA:
    """
    This is an implementation of kernel principal component analysis.

    Inputs:
    - n_components = number of components to retain in projection
    - kernel = kernel function (linear, polynomial or RBF)
    - gamma = gamma parameter for polynomial kernel and RBF (used to
              compute std. deviation)
    - degree = degree of polynomial kernel
    - coef0 = 0-degree parameter for polynomial kernel
    """

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
        """
        This method can be used to fit the PCA and project data into it
        (retaining only first 'n_components' PCs)
        """
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


class TwoNN:
    """
    This is an implementation of the twoNN algorithm for intrinsic
    dimensionality estimation.
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


class FCIEstimator:
    """
    This is an implementation of (global) full correlation integral
    estimator for intrinsic dimension.
    Data for fitting are expected to be passed as a numpy array
    with one row per datapoint and one column per feature.

    NOTICE: we didn't use the rescaling parameter r_s of the
    original paper, as it turned out to be redundant when
    normalized the last argument of the (2,1)-hypergeometric
    function in the FCI estimator.
    """

    def __init__(self):
        super(FCIEstimator, self).__init__()

        self.d = None  # intrinsic dimension
        self.d_std = None  # standard deviation of parameter estimations

    def fit(self, data: np.ndarray, r: np.ndarray):
        """
        r: values of radius to be used for fitting
        """

        # center and normalize data
        data = preprocess(data)

        # computing true density values
        ngbrs_density = compute_empirical_FCI(data, r)

        # fit model
        initial_guess = data.shape[1] / 2.  # initial guess for free parameter
        bounds = (0., data.shape[1])  # bounds for free parameter

        params_opt, params_covariance = curve_fit(
                estimate_FCI,
                r,
                ngbrs_density,
                p0=initial_guess,
                bounds=bounds,
        )

        # get optimal values for parameters with st. dev.
        self.d = params_opt[0]
        self.d += 1  # to compensate degree of freedom loss in normalization
        self.d_std = np.sqrt(np.diag(params_covariance))[0]

    def return_estimate(self):
        return self.d, self.d_std
