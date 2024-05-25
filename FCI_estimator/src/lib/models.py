import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from .preprocessing import preprocess
from .density import compute_empirical_FCI, estimate_FCI


class FCI_estimator:
    """
    This class can be used to instantiate a full correlation
    integral estimator for intrinsic dimension.
    Data for fitting are expected to be passed as a numpy array
    with one row per datapoint and one column per feature.
    """

    def __init__(self):
        self.r_s = None  # scaling factor for cutoff distance
        self.d = None  # intrinsic dimension
        self.params_std = None  # standard deviation of parameter estimations

    def fit(self, data: np.ndarray, r: np.ndarray):
        """
        r: values of radius to be used for fitting
        """

        # center and normalize data
        data = preprocess(data)

        # computing true density values
        ngbrs_density = compute_empirical_FCI(data, r)

        #print(f"computed densities: {ngbrs_density}")

        # fit model
        initial_guess = [np.mean(pdist(data)), np.sqrt(data.shape[1])]  # initial guess for parameters
        bounds = (0., [np.inf, data.shape[1]])  # bounds for parameters

        print(f"average distance between datapoints: {np.mean(pdist(data))}")

        params_optimal, params_covariance = curve_fit(
                estimate_FCI,
                r,
                ngbrs_density,
                p0=initial_guess,
                bounds=bounds,
        )

        # get optimal values for parameters with st. dev.
        self.r_s, self.d = params_optimal
        self.d += 1  # to compensate degree of freedom loss in normalization
        self.params_std = np.sqrt(np.diag(params_covariance))

        #print(f"condition number: {np.linalg.cond(params_covariance)}")

    def return_estimate(self):
        return self.r_s, self.d, self.params_std
