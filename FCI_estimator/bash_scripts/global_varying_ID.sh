#!/bin/bash


echo "Running global estimator on datasets D, G and H, for varying intrinsic dimension..."
echo "-- dataset size is set to 700 for all datasets"
echo "-- embedding dimension is set to 1024 for all datasets"

cd ../src/

for ((id = 2; id <= 1024; id *= 2))
do
    python3 make_datasets.py -N 700 -D 1500 -d $id

    python3 fit.py --mode "g" --results_path "../results/global_varying_ID/${id}.json"
done

cd ../bash_scripts/
