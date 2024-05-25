import numpy as np
from scipy.constants import pi
from scipy.special import gamma, hyp2f1
from scipy.spatial.distance import pdist, squareform


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


def estimate_FCI(r: np.ndarray, r_s: float, d: float):
    """
    This function estimates the correlation integral for a given cutoff
    radius r, rescaled of r_s in dimension d.
    It is used in the non linear fit for the model.
    """

    # precompute needed quantities for CI estimation
    solid = 2. * pi**(0.5 * d) / gamma(0.5 * d)  # solid d-dimensional angle
    solid_1 = 2. * pi**(0.5 * (d-1)) / gamma(0.5 * (d-1))  # solid (d-1)-dimensional angle
    angle_ratio = 0.5 * solid_1 / solid

    # rescale r
    r_rescaled = r / r_s

    # precompute and normalize last arg (cannot be larger than 1)
    last_arg = (r_rescaled**2 - 2) ** 2
    last_arg /= np.max(last_arg)

    # compute (2,1)-hypergeometric function
    with np.errstate(divide='ignore', invalid='ignore'):
        hypergeom = hyp2f1(0.5, 1. - 0.5 * d, 1.5, last_arg)
        edge_cases_count = np.sum(np.isnan(hypergeom) | np.isinf(hypergeom))
        hypergeom[np.isinf(hypergeom)] = 0  # handle edge cases

    #print(f"fractioin of edge cases: {edge_cases_count / hypergeom.shape[0]}")
    #print(f"last_arg: {last_arg}")
    #print(f"hypergeom: {hypergeom}")

    return 0.5 + angle_ratio * (r_rescaled**2 - 2) * hypergeom
