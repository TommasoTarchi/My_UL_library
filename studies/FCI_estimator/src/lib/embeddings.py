import numpy as np
import random


# used in the following function
def get_rotation_matrix(theta: np.ndarray, n_1: np.ndarray, n_2: np.ndarray) -> np.ndarray:
    """
    Taken from https://github.com/davidmickisch/torch-rot/blob/main/torch_rot/rotations.py and
    moved from pytorch to numpy
    This method returns a rotation matrix which rotates any vector
    in the 2 dimensional plane spanned by
    @n1 and @n2 an angle @theta. The vectors @n1 and @n2 have to be orthogonal.
    Inspired by
    https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    :param @n1: first vector spanning 2-d rotation plane, needs to be orthogonal to @n2
    :param @n2: second vector spanning 2-d rotation plane, needs to be orthogonal to @n1
    :param @theta: rotation angle
    :returns : rotation matrix
    """

    dim = len(n_1)
    assert len(n_1) == len(n_2)
    assert (np.abs(n_1.dot(n_2)) < 1e-4)

    return (np.eye(dim) +
            (np.outer(n_2, n_1) - np.outer(n_1, n_2)) * np.sin(theta) +
            (np.outer(n_1, n_1) + np.outer(n_2, n_2)) * (np.cos(theta) - 1))


# used in linear embedding
def rotate_random(data: np.ndarray, orthog_method='GS'):
    """
    This method takes a matrix with data as input and returns it rotated of a random angle
    along a random hyperplane (the hyperplane is chosen as the span of a random vector and
    a vector orthonormal to the first one).
    Data are expected to be in the form of a matrix with rows representing datapoints and
    columns representing features.
    'orthog_method' is the orthoginalization method; can be:
    1. "GS": Gram-Schmidt (default),
    2. "flip": second vector built with all components equal to zero but two, which are
    flipped from the first vector and one of them changed of sign.
    """

    dim = data.shape[1]

    # extract random angle
    angle = np.random.uniform(0, 2*np.pi, 1)

    # extract random hyperplane ("flip elements" or Gram-Schmidt approach)
    if orthog_method == "GS":
        vector1 = np.random.randn(dim)  # first vector spanning hyperplane
        vector2 = np.random.randn(dim)  # second vector spanning hyperplane
        vector1 /= np.linalg.norm(vector1)
        vector2 -= vector1 * np.dot(vector1, vector2) / np.dot(vector1, vector1)
        vector2 /= np.linalg.norm(vector2)
    elif orthog_method == "flip":
        vector1 = np.random.randn(dim)  # first vector spanning hyperplane
        vector2 = np.zeros_like(vector1)  # init second vector spanning hyperplane
        flip_indexes = random.sample(range(dim), 2)  # choose values to flip
        vector2[flip_indexes[0]] = -vector1[flip_indexes[1]]  # flip first value (changing sign)
        vector2[flip_indexes[1]] = vector1[flip_indexes[0]]  # flip second value
        vector1 /= np.linalg.norm(vector1)  # normalize
        vector2 /= np.linalg.norm(vector2)  # normalize
    else:
        raise ValueError("Orthogonalization method not valid: choose among 'GS' and 'flip'")

    rotation_mat = get_rotation_matrix(angle, vector1, vector2)

    return np.dot(data, rotation_mat.T)


# used in the following embedding functions
def increase_dimension(data: np.ndarray, target_dimension: int):
    """
    This method simply adds zeros to data to match target
    dimension (i.e. maps data to higher space).
    Data are expected to be in the form of a matrix in which
    each row is a datapoint and each column a feature.
    """

    if target_dimension < data.shape[1]:
        raise ValueError("Target dimension must be larger than data dimension")

    padding = np.zeros((data.shape[0], target_dimension-data.shape[1]), dtype=data.dtype)

    return np.hstack((data, padding))


# linear embedding
def embed_linear(data: np.ndarray, embedding_dim: int, orthog_method: str):

    # map data to higher-dimensnional space
    data_embedded = increase_dimension(data, embedding_dim)

    # rotate data
    data_embedded = rotate_random(data_embedded, orthog_method)

    # remove residual values from rotation
    data_embedded[data_embedded < 1e-8] = 0.0

    return data_embedded


# embedding for dataset C (embedding dimension is twice the
# original one)
def embed_C(data: np.ndarray):

    size = data.shape[0]
    dim = data.shape[1]
    embedding_dim = dim * 2

    # embed data to higher-dimensnional space
    data_embedded = np.empty((size, embedding_dim), dtype=data.dtype)
    for feature in range(dim-1):
        data_embedded[:, feature*2] = data[:, feature+1] * np.cos(data[:, feature])
        data_embedded[:, feature*2+1] = data[:, feature+1] * np.sin(data[:, feature])
    data_embedded[:, embedding_dim-2] = data[:, 0] * np.cos(data[:, dim-1])
    data_embedded[:, embedding_dim-1] = data[:, 0] * np.sin(data[:, dim-1])

    # remove residual values from rotation
    data_embedded[data_embedded < 1e-8] = 0.0

    return data_embedded


# embedding for dataset SR (original dimension is 2 and embedding
# dimension is 3)
def embed_SR(data: np.ndarray):

    # fixed original dimension to 2 and fixed embedding dimension to 3
    if data.shape[1] != 2:
        raise ValueError("Dataset dimension must be equal to 2")

    size = data.shape[0]

    # embed data to higher-dimensnional space
    data_embedded = np.empty((size, 3), dtype=data.dtype)
    data_embedded[:, 0] = data[:, 0] * np.cos(2*np.pi*data[:, 1])
    data_embedded[:, 1] = data[:, 1]
    data_embedded[:, 2] = data[:, 0] * np.sin(2*np.pi*data[:, 1])

    return data_embedded


# adds Gaussian noise to dataset
def add_gauss_noise(data: np.ndarray, std_dev: int):

    noise = np.random.normal(loc=0.0, scale=std_dev, size=data.shape)

    return data + noise
