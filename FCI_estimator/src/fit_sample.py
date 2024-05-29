import numpy as np
import json

from lib.models import GlobalFCIEstimator


if __name__ == "__main__":

    # aggiungere come argomento da command line r
    r = np.linspace(0.8, 2., 50)

    # read and preprocess datasets
    data_names = ['D', 'G', 'H']
    datasets = []
    for name in data_names:
        datasets.append(np.load('../datasets/' + name + '.npy'))

    # fit model for all datasets
    results = {}
    for data, name in zip(datasets, data_names):
        print(f"fitting model on dataset {name}...")
        estimator = GlobalFCIEstimator()
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
    with open("../results/sample_results.json", 'w') as file:
        json.dump(results_for_json, file, indent=4)

    print(results_for_json)