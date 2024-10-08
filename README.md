# My unsupervised learning library

In this repository you can find my personal implementation of several unsupervised
learning algorithms.

See [here](#how-to-use-the-library) for library usage instructions.

An [entire directory](FCI_estimator) is dedicated to implementation and testing of
the algorithm described in [this paper][link1] by Vittorio Erba, Marco Gherardi and
Pietro Rotondo.


## What you will find in this repository

This repo contains:
- This README file
- `UL_lib/`: unsupervised learning library (see [here](#how-to-use-the-library) for use)
- `studies/FCI_estimator/`: directory containing codes for testing of FCI estimator by
Erba, Gherardi and Rotondo (see [here](#study-on-fci-estimator) for more)
- `environment.yaml`: YAML file for building conda environment
- `requirements.txt`: requirements file


## How to use the library

The library contains the following modules:
- `clustering.py`, implementing:
  - `kmeans`: k-means
  - `FuzzyCMeans`: fuzzy c-means
  - `SpectralClustering`: spectral clustering
  - `DensPeakClustering`: density peak clustering
- `density_estimation.py`, implementing:
  - `HistEstimator`: histogram density estimator
  - `GaussKDEstimator`: kernel density estimator with Gaussian kernel
- `dimensionality_reduction.py`, implementing:
  - `PCA`: principal component analysis
  - `Isomap`: isomap (**Notice**: this method still has to be revised,
    we do not guarantee it to work properly)
  - `KernelPCA`: kernel principal component analysis
  - `TwoNN`: two NN
  - `FCIEstimator`: full correlation integral-based intrinsic dimension estimator
- `metrics.py`, implementing:
  - `compute_MI`: mutual information calculator
  - `compute_NMI`: normalized mutual information calculator
  - `compute_FRatio`: F-ratio score calculator

To use these methods you can follow these simple steps:

1. Clone this repository:
   ````
   git@github.com:TommasoTarchi/My_UL_library.git
   ````

2. Prepare the environment:

    - If using conda, substitute your desired environment's name to `<your_env_name>` in the first line
      of `environment.yaml`, and build the conda environment using:
      ````
      $ conda env create -f environment.yaml
      ````

    - If using Pip, just do:
      ````
      $ pip install -r requirements.txt
      ````

3. Copy the [UL_lib directory](UL_lib) in your working directory.

4. use the desired classes/functions by importing them into your python script:
    ````
    from UL_lib.<module_name> import <function/class name>
    ````
    For instance, if you want to use the k-means algorithm you can use:
    ````
    from UL_lib.clustering import kmeans
    ````


## Study on FCI estimator

`studies/FCI_estimator/` directory contains:
- `bash_scripts/`: bash scripts for running tests
- `datasets/`: datasets used to test the algorithm
- `src/`: codes for implementation and testing of the algorithm
- `results/`: directory containing results of tests
- `FCI_estimator-Presentation.pdf`: presentation of implementation and results

To reproduce the tests it is sufficient to navigate to the `FCI_estimator/bash_scripts/`
directory and run the bash scripts. Parameters in the scripts can be adjusted to
investigate the desired parameter configurations.

For details about implementation and testing of the algorithm (in particular for numerical
"subtleties" in implementation) see [this presentation](studies/FCI_estimator/FCI_estimator-Presentation.pdf).


## References

The intrinsic dimension estimator for undersampled data implemented and tested in
`FCI_estimator/`, was taken from the paper:

Erba, V., Gherardi, M. & Rotondo, P. Intrinsic dimension estimation for locally undersampled
data. Sci Rep 9, 17133 (2019). [link to paper][link1].






[link1]: https://www.nature.com/articles/s41598-019-53549-9
