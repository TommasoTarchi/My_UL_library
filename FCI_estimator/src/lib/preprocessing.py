import numpy as np


def preprocess(data: np.ndarray):
    """
    This function centers and normalize a dataset.
    The dataset is expected to be passed in the form of a numpy array
    with one row per datapoint and one column per feature.
    """

    # center data
    mass_center = np.mean(data, axis=0, dtype=data.dtype)
    data_centered = data - mass_center

    # normalize data
    modules = np.linalg.norm(data_centered, axis=1)
    modules[modules == 0.0] = 1.0  # possible null datapoint
    data_normalized = data_centered / modules.reshape(-1, 1)

    return data_normalized
