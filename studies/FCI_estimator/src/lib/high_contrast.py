import numpy as np


def generate_images(dset_size, l, n):
    """
    This function generates a dataset of l*l high contrast images,
    each one containing n blobs with random coordinates and shape
    (each image is built by generating n single-blob images and
    summing them all together).
    For mathematical details about the way the images are created
    see original paper by Erba, Gherardi and Rotondo.
    Pixels with value below 0.01 are manually set to 0, to increase
    contrast.
    """

    # generate blob parameters
    #
    # (we generate one set of parameters for each blob,
    # n*dset_size sets in total)
    dx = l/2 + np.random.uniform(-20.0, 20.0, n*dset_size)  # horizontal translation
    dy = -l/2 + np.random.uniform(-20.0, 20.0, n*dset_size)  # vertical translation
    s = np.random.uniform(1.0, 3.0, n*dset_size)  # size
    e = np.random.uniform(5.0, 10.0, n*dset_size)  # eccenticity
    theta = np.random.uniform(-np.pi/2.0, np.pi/2.0, n*dset_size)  # angle of major axis

    # compute pixels values in single-blob images
    v = np.empty((dset_size*n, l, l), dtype=np.float64)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    for i in range(l):
        for j in range(l):
            a = cos_theta * (j-dx) + sin_theta * (i+dy)
            b = -sin_theta * (j-dx) + cos_theta * (i+dy)
            v[:, i, j] = 1.0 - np.sqrt((a**2 + e * b**2) / (1 + e)) / s
    v[v < 0.0] = 0.0  # discard negative pixel values
    v /= np.max(v)  # normalize images

    # compute n-blobs images by summing single-blob images
    v_reshaped = v.reshape((dset_size, n, l, l))
    data = np.sum(v_reshaped, axis=1)

    # normalize pixel values
    for i in range(dset_size):
        data[i] /= np.max(data[i])

    # increase contrast
    data[data < 0.01] = 0.0

    return data
