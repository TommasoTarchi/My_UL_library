import argparse
import numpy as np

from lib.embeddings import embed_linear, embed_C, embed_SR, add_gauss_noise
from lib.high_contrast import generate_images


if __name__ == "__main__":

    # get command line arguments
    parser = argparse.ArgumentParser(description="This program creates a simple dataset to test intrinsic dimension estimation")
    parser.add_argument('-N', type=int, default=1000, help='Size of generated dataset')
    parser.add_argument('-d', type=int, default=15, help='Intrinsic dimension')
    parser.add_argument('-D', type=int, default=60, help='Embedding dimension (must be larger than d)')
    parser.add_argument('-n', type=int, default=3, help='Number of blobs in images of high-contrast dataset')
    parser.add_argument('--orthog_method', type=str, default="GS", help='Orthogonalization method for building random rotation hyperplane (default Gram-Schmidt)')
    parser.add_argument('--noise_std_dev', type=float, default=0.0, help='Standard deviation of Gaussian noise added to datsets D, G and H')

    args = parser.parse_args()

    N = args.N  # size of datasets
    d = args.d  # intrinsic dimension of data
    D = args.D  # dimension of the embedding space
    num_blobs = args.n  # number of blobs in high-contrast images
    orthog_method = args.orthog_method  # orthogonalization method for rotation
    noise_std_dev = args.noise_std_dev  # standard deviation of Gaussian noise

    if D < d:
        raise ValueError("Embedding dimension must be larger than intrinsic dimension")

    # sample data for simpler datasets (to be embedded in
    # higher-dimensional space)
    #
    # (names of datasets are the same as in "description
    # of the dataset" in original paper)
    dset_D = np.random.choice(2, (N, d))  # uniform binary
    dset_G = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)  # standard multivariate normal
    dset_H = np.random.rand(N, d)  # uniform (values in [0, 1)^d)
    dset_C = np.random.rand(N, d) * 2 * np.pi  # uniform (values in [0, 2pi)^d)
    dset_SR = np.random.rand(N, 2)  # uniform (values in [0, 1)^d) (notice: fixed intrinsic dimension)

    # embed data
    dset_D = embed_linear(dset_D, embedding_dim=D, orthog_method=orthog_method)
    dset_G = embed_linear(dset_G, embedding_dim=D, orthog_method=orthog_method)
    dset_H = embed_linear(dset_H, embedding_dim=D, orthog_method=orthog_method)
    dset_C = embed_C(dset_C)
    dset_SR = embed_SR(dset_SR)

    # build high-contrast images dataset
    image_size = 81  # length of images' side (fixed to same value as in the paper)
    dset_B = generate_images(N, image_size, num_blobs)

    # optionally add noise
    if noise_std_dev:
        dset_D = add_gauss_noise(dset_D, noise_std_dev)
        dset_G = add_gauss_noise(dset_G, noise_std_dev)
        dset_H = add_gauss_noise(dset_H, noise_std_dev)

    # save datasets to binary file
    np.save('../datasets/D.npy', dset_D)
    np.save('../datasets/G.npy', dset_G)
    np.save('../datasets/H.npy', dset_H)
    np.save('../datasets/C.npy', dset_C)
    np.save('../datasets/SR.npy', dset_SR)
    np.save('../datasets/B.npy', dset_B)
