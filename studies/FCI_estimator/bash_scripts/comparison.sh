#!/bin/bash


echo "Running FCI and TwoNN estimators on dataset G, for varying intrinsic dimension..."
echo "-- dataset size is set to 700 for all datasets"
echo "-- embedding dimension is set to 1024 for all datasets"

cd ../src/

for ((id = 2; id <= 1024; id *= 2))
do
    python3 make_datasets.py -N 700 -D 1500 -d $id --noise_std_dev 0.0

    python3 fit_comparison.py --results_path "../results/comparison/${id}.json"
done

cd ../bash_scripts/
