import argparse
import numpy as np
import json
from dadapy.data import Data

from lib.models import GlobalFCIEstimator


if __name__ == "__main__":

    # get command line arguments
    parser = argparse.ArgumentParser(description="This program runs comparison between towNN (dadapy implementation) and FCI estimators on dataset H")
    parser.add_argument('--results_path', type=str, help='Complete path to file to store results (name of the JSON file with extention included)')

    args = parser.parse_args()

    results_path = args.results_path  # path to file to store results

    # check validity of datafile
    if not results_path:
        raise ValueError("A path to JSON file to store results must be passed")
    if results_path[-5:] != '.json':
        raise ValueError("Datafile must be a valid JSON file")

    # radii used in the FCI estimator algorithm
    r = np.linspace(0.8, 2., 50)

    # read dataset
    data = np.load('../datasets/G.npy')

    results = {}

    # fit FCI estimator
    print(f"fitting FCI estimator...")
    FCI_estimator = GlobalFCIEstimator()
    FCI_estimator.fit(data, r)
    results['FCI'] = FCI_estimator.return_estimate()

    # fit twoNN estimator
    print(f"fitting towNN estimator...")
    data_for_twoNN = Data(data)
    results['TwoNN'] = data_for_twoNN.compute_id_2NN()

    # save results to json
    results_for_json = {
        model: {
            'optimal': values[0]
        }
        for model, values in results.items()
    }
    with open(results_path, 'w') as file:
        json.dump(results_for_json, file, indent=4)
