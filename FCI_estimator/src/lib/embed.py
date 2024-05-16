# this file contains different embedding maps needed to build
# the datasets


import numpy as np

from .rotate import rotate_random


# used in the following embedding functions
def increase_dimension(data: np.ndarray, target_dimension: int):
    """
    This method simply adds zeros to data to match target
    dimension (i.e. maps data to higher space)
    data are expected to be in the form of a matrix in which
    each row is a datapoint and each column a feature
    """

    if target_dimension < data.shape[1]:
        raise ValueError("Target dimension must be larger than data dimension")

    padding = np.zeros((data.shape[0], target_dimension-data.shape[1]), dtype=data.dtype)

    return np.hstack((data, padding))


# ATTENZIONE A N NELLE DEFINIZIONI DEI DATASET NEL PAPER


def embed_linear(data, embedding_dim):

    # map data to higher-dimensnional space
    data_embedded = increase_dimension(data, embedding_dim)

    # rotate data
    data_embedded = rotate_random(data_embedded)

    return data_embedded


def embed_C(data):

    # embed data to higher-dimensnional space
    data_embedded = ...

    return data_embedded


def embed_SR(data):

    if data.shape[1] != 2:
        raise ValueError("Dataset dimension must be equal to 2")

    # embed data to higher-dimensnional space
    data_embedded = ...

    return data_embedded
