import numpy as np
from scipy.optimize import curve_fit

from .preprocessing import preprocess
from .density import compute_density, estimate_CI


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
        data = preprocess(data)

        # computing true density values
        ngbrs_density = compute_density(data, r)

        # fit model
        params_optimal, params_covariance = curve_fit(estimate_CI, r, ngbrs_density)

        # get optimal values for parameters with st. dev.
        self.r_s, self.d = params_optimal
        self.params_std = np.sqrt(np.diag(params_covariance))

    def return_estimate(self):
        return (self.r_s, self.d, self.params_std)
