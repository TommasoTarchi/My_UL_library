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
    NOTICE: we didn't use the rescaling parameter r_s of the
    original paper, as it turned out to be redundant when
    normalized the last argument of the (2,1)-hypergeometric
    function in the FCI estimator.
    """

    def __init__(self):
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
        initial_guess = data.shape[1] / 2.  # initial guess for free parameter
        bounds = (0., data.shape[1])  # bounds for free parameter

        #print(f"average distance between datapoints: {np.mean(pdist(data))}")

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
