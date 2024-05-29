import argparse
import numpy as np
import json

from lib.models import GlobalFCIEstimator, MultiscaleFCIEstimator


if __name__ == "__main__":

    # get command line arguments
    parser = argparse.ArgumentParser(description="This program runs global or multiscale FCI estimation of intrinsic dimension")
    parser.add_argument('--mode', type=str, default='g', choices=('g', 'm'), help='Whether to perform global ("g") or multiscale ("m") FCI estimation')
    parser.add_argument('--datapath', type=str, help='Complete path to datafile (name of the JSON file with extention included)')

    args = parser.parse_args()

    mode = args.mode  # whether to perform global or multiscale FCI estimation
    datapath = args.datapath  # path to datafile

    # check validity of datafile
    if not datapath:
        raise ValueError("A path to JSON file to store results must be passed")
    if datapath[-5:] != '.json':
        raise ValueError("Datafile must be a valid JSON file")

    # radii used in the algorithm
    r = np.linspace(0.8, 2., 50)

    # read and preprocess datasets
    if mode == 'g':
        data_names = ['D', 'G', 'H']
    elif mode == 'm':
        data_names = ['B']
    datasets = []
    for name in data_names:
        datasets.append(np.load('../datasets/' + name + '.npy'))

    # fit model for all datasets
    results = {}
    for data, name in zip(datasets, data_names):
        print(f"fitting model on dataset {name}...")
        if mode == 'g':
            estimator = GlobalFCIEstimator()
        elif mode == 'm':
            estimator = MultiscaleFCIEstimator()
        estimator.fit(data, r)
        results[name] = estimator.return_estimate()

    # save results to json
    results_for_json = {
        dataset: {
            'optimal': values[0],
            'std_dev': values[1]
        }
        for dataset, values in results.items()
    }
    with open(datapath, 'w') as file:
        json.dump(results_for_json, file, indent=4)
