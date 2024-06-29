import numpy as np
from scipy.constants import pi
from scipy.special import gamma, hyp2f1
from scipy.spatial.distance import pdist, squareform
import mpmath


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
