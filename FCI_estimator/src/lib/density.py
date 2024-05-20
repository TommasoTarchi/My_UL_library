import numpy as np
from scipy.constants import pi
from scipy.special import gamma
from scipy.special import hyp2f1


def compute_density(data: np.ndarray, r: np.ndarray):
    """
    This function computes the density of neighbours (also known as
    "correlation integral" in the literature) of a dataset, for a
    given list of cutoff distances (passed in the form of a numpy
    array r).
    Data are expected to be passed in the form of a numpy array with
    one row per datapoint and one column per feature.
    """

    dset_size = data.shape[0]

    # compute normalization factor
    norm = 2 / dset_size / (dset_size-1)

    # compute distances between all couples of datapoints
    # (triangular matrix, with all zeros on the lower triangle)
    distances = np.zeros((dset_size, dset_size), dtype=np.float64)
    for i in range(dset_size):
        for j in range(i+1, dset_size):
            distances[i, j] = np.linalg.norm(data[i] - data[j])

    # compute number of neighbours within cutoff distances
    accumulate_distance = np.zeros_like(r, dtype=np.float64)
    for i in range(r.shape[0]):
        accumulate_distance[i] += np.sum(distances <= r[i])

    print(accumulate_distance)
    print(norm)

    return accumulate_distance * norm


def estimate_CI(r: float, r_s: float, d: int):
    """
    This function estimates the correlation integral for a given cutoff
    radius r, rescaled of r_s in dimension d.
    It is used in the non linear fit for the model.
    """

    # precompute needed quantities for CI estimation
    solid = 2. * pi**(0.5 * d) / gamma(0.5 * d)  # solid d-dimensional angle
    solid_1 = 2. * pi**(0.5 * (d-1)) / gamma(0.5 * (d-1))  # solid (d-1)-dimensional angle
    angle_ratio = 0.5 * solid_1 / solid
    second_arg = 1. - 0.5 * d  # second argument for hypergeometric function

    # rescale r
    r /= r_s

    # compute (2,1)-hypergeometric function
    hypergeom = hyp2f1(0.5, second_arg, 1.5, (r**2 - 2) ** 2)

    return 0.5 + angle_ratio * (r**2 - 2) * hypergeom
