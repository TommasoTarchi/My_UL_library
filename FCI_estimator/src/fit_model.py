import numpy as np

from lib.preprocessing import preprocess
from lib.models import FCI_estimator


if __name__ == "__main__":

    # aggiungere come argomento da command line r

    # read and preprocess datasets
    data_names = ['D', 'G', 'H', 'C', 'SR', 'B']
    datasets = []
    for name in data_names:
        datasets.append(preprocess(np.load('../datasets/' + name + '.npy')))

    # fit model for all datasets
    results = {}
    for data, name in zip(datasets, data_names):
        estimator = FCI_estimator()
        r = np.linspace(0.8, 2, 30)
        estimator.fit(data, r)
        results[name] = estimator.return_estimate()[1]

    # print results
    print(results)
