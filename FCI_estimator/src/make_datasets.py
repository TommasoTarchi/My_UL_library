import argparse
import numpy as np

from lib.embed import embed_linear, embed_C, embed_SR


if __name__ == "__main__":

    # get command line arguments
    parser = argparse.ArgumentParser(description="This program creates a simple dataset to test intrinsic dimension estimation")
    parser.add_argument('-N', type=int, default=1000, help='Size of generated dataset')
    parser.add_argument('-d', type=int, default=10, help='Intrinsic dimension')
    parser.add_argument('-D', type=int, default=15, help='Embedding dimension (must be larger than d)')

    args = parser.parse_args()

    N = args.N  # size of datasets
    d = args.d  # intrinsic dimension of data
    D = args.D  # dimension of the embedding space

    if D < d:
        raise ValueError("Embedding dimension must be larger than intrinsic dimension")

    # sample data (to be embedded in higher-dimensional space)
    #
    # (names of datasets are the same as in "description
    # of the dataset" in original paper)
    dset_D = np.random.choice(2, (N, d))  # uniform binary
    dset_G = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)  # standard multivariate normal
    dset_H = np.random.rand(N, d)  # uniform (values in [0, 1)^d)
    dset_C = np.random.rand(N, d) * 2 * np.pi  # uniform (values in [0, 2pi)^d)
    dset_SR = np.random.rand(N, 2)  # uniform (values in [0, 1)^d) (notice: fixed intrinsic dimension)

    # embed data
    dset_D = embed_linear(dset_D, embedding_dim=D)
    dset_G = embed_linear(dset_G, embedding_dim=D)
    dset_H = embed_linear(dset_H, embedding_dim=D)
    dset_C = embed_C(dset_C)
    dset_SR = embed_SR(dset_SR)

    # build high-contrast images dataset
