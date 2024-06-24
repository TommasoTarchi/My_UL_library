# My unsupervised learning library

In this repository you can find my personal implementation of several unsupervised
learning algorithms.

An [entire directory](FCI_estimator) is dedicated to implementation and testing of
the algorithm described in [this paper][link1] by Vittorio Erba, Marco Gherardi and
Pietro Rotondo.

If you want to use the methods (FCI estimator excluded) you can find more instructions
[here](#ref1).


## What you will find in this repository

This repo contains:
- This README file
- `UL_lib/`: unsupervised learning library (see [here](#ref1) for more)
- `FCI_estimator/`: directory containing codes for FCI estimator by Erba, Gherardi
  and Rotondo (see [here](#ref2) for more)
- `environment.yml`: requirements file for conda environment


<a name="ref1">
</a>

### UL\_lib

This directory contains the following modules:
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
- `metrics.py`, implementing:
  - `compute_MI`: mutual information calculator
  - `compute_NMI`: normalized mutual information calculator
  - `compute_FRatio`: F-ratio score calculator

To use these methods you can follow these simple steps:

1. Download [this directory](UL_lib) in your working directory

2. Substitute your environment's name to `<your_env_name>` in the first
line of `environment.yml`, and build the conda environment using:

````
$ conda env create -f environment.yml
````

3. (Optional) remove `environment.yml`, to make the library cleaner

4. use the desired classes/functions by importing them into your python script:

````
from UL_lib.<module_name> import <function/class name>
````

For instance, if you want to use the k-means algorithm you can use:

````
from UL_lib.clustering import kmeans
````


<a name="ref2">
</a>

### FCI\_estimator

This directory contains:
- `bash_scripts/`: bash scripts for running tests
- `datasets/`: datasets used to test the algorithm
- `src/`: codes for implementation and testing of the algorithm
- `results/`: directory containing results of tests
- `FCI_estimator-Presentation.pdf`: presentation of implementation and results

To reproduce the tests it is sufficient to navigate to the `FCI_estimator/bash_scripts/`
directory and run the bash scripts. Parameters in the scripts can be adjusted to
investigate the desired parameter configurations.

For details about implementation and testing of the algorithm (in particular for numerical
"subtleties" in implementation) see [this presentation](FCI_estimator/FCI_estimator-Presentation.pdf).


## References

The intrinsic dimension estimator for undersampled data implemented and tested in
`FCI_estimator/`, was taken from the paper:

Erba, V., Gherardi, M. & Rotondo, P. Intrinsic dimension estimation for locally undersampled
data. Sci Rep 9, 17133 (2019). [link to paper][link1].






[link1]: https://www.nature.com/articles/s41598-019-53549-9
