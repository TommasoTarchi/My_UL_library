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
    assert (n_1.dot(n_2).abs() < 1e-4)

    return (np.eye(dim) +
            (np.outer(n_2, n_1) - np.outer(n_1, n_2)) * np.sin(theta) +
            (np.outer(n_1, n_1) + np.outer(n_2, n_2)) * (np.cos(theta) - 1))


# used in embed.py
def rotate_random(data: np.ndarray):
    """
    This method takes a matrix with data as input and returns it rotated of a random angle
    along a random hyperplane (the hyperplane is chosen as the span of a random vector and
    a vector orthonormal to the first one, built by taking a vector with all zeros but two
    elements flipped from the first one (one changed of sign))
    data should be passed in the form of a matrix with rows representing datapoints and
    columns representing features
    """

    dim = data.shape[1]

    # extract random angle and hyperplane
    angle = np.random.uniform(0, 2*np.pi, 1)
    vector1 = np.random.uniform(-1, 1, dim)  # first vector spanning hyperplane
    vector2 = np.zeros_like(vector1)  # init second vector spanning hyperplane
    flip_indexes = random.sample(range(dim), 2)  # choose values to flip
    vector2[flip_indexes[0]] = -vector1[flip_indexes[1]]  # flip first value (changing sign)
    vector2[flip_indexes[1]] = vector1[flip_indexes[0]]  # flip second value
    vector1 /= np.linalg.norm(vector1)  # normalize
    vector2 /= np.linalg.norm(vector2)  # normalize

    # compute rotation matrix
    rotation_mat = get_rotation_matrix(angle, vector1, vector2)

    return np.dot(data, rotation_mat.T)
