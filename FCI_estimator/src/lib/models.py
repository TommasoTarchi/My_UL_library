import numpy as np
from scipy.optimize import curve_fit
import random
from sklearn.neighbors import NearestNeighbors

from .preprocessing import preprocess
from .density import compute_empirical_FCI, estimate_FCI


class GlobalFCIEstimator:
    """
    This class can be used to instantiate a (global) full
    correlation integral estimator for intrinsic dimension.
    Data for fitting are expected to be passed as a numpy array
    with one row per datapoint and one column per feature.
    NOTICE: we didn't use the rescaling parameter r_s of the
    original paper, as it turned out to be redundant when
    normalized the last argument of the (2,1)-hypergeometric
    function in the FCI estimator.
    """

    def __init__(self):
        super(GlobalFCIEstimator, self).__init__()

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


class MultiscaleFCIEstimator:
    """
    This class can be used to instantiate a multiscale full
    correlation integral estimator for intrinsic dimension.
    To achieve multiscale estimation, the FCI estimator is
    applied at multiple scales (parameter r_c), and the
    smallest obtained intrinsic dimension is selected as the
    final result.
    The application of the single FCI estimators is the same
    as in the previous class.
    """

    def __init__(self):
        super(MultiscaleFCIEstimator, self).__init__()

        self.d = []
        self.d_std = []

    def fit(self, data: np.ndarray, r: np.ndarray, NN_res: int = 20):
        """
        r: values of radius to be used for fitting
        NN_res: resolution of the vector of numbers of nearest
                neighbors
        """

        # center and normalize data
        data = preprocess(data)

        # select random datapoint as center
        center_index = random.randint(0, data.shape[0] - 1)

        # apply FCI estimator for required numbers of neighbors
        for k in range(10, data.shape[0], NN_res):

            # select k nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
            _, indices = nbrs.kneighbors(data[center_index].reshape(1, -1))
            indices = indices.flatten()
            data_NN = data[indices, :]

            # computing true density values
            ngbrs_density = compute_empirical_FCI(data_NN, r)

            # fit model
            initial_guess = data_NN.shape[1] / 2.  # initial guess for free parameter
            bounds = (0., data_NN.shape[1])  # bounds for free parameter

            params_opt, params_covariance = curve_fit(
                    estimate_FCI,
                    r,
                    ngbrs_density,
                    p0=initial_guess,
                    bounds=bounds,
            )

            # get optimal values for parameters with st. dev.
            self.d.append(params_opt[0])
            self.d_std.append(np.sqrt(np.diag(params_covariance))[0])

    def return_estimate(self):
        # select smaller ID as optimal
        index_opt = np.argmin(self.d)

        # return optimal value (adjusted for loss of degree of freedom)
        # with standard deviation
        return self.d[index_opt]+1, self.d_std[index_opt]


class TwoNN:
    """
    (Max-likelihood) twoNN estimator for intrinsic dimension.
    (Here used only as a term of comparison for the FCI estimator).
    """

    def __init__(self):
        self.d = None
        self.r = None

    def fit(self, data):
        N = data.shape[0]

        # compute nearest neighbors
        ngbrs = NearestNeighbors(n_neighbors=4, algorithm='auto')
        ngbrs.fit(data)

        # compute distance ratios
        ngbrs_distances, _ = ngbrs.kneighbors(data)
        indexes = list(range(N))
        for i in range(N):
            if ngbrs_distances[i, 1] == 0:
                indexes.remove(i)
        mu = ngbrs_distances[indexes, 2] / ngbrs_distances[indexes, 1]

        # estimate d (through max-likelihood)
        self.d = mu.shape[0] / np.sum(np.log(mu))

        # average distance from nearest neighbor
        self.r = np.mean(ngbrs_distances[indexes, 1])

    def return_estimate(self):
        return self.d, self.r
