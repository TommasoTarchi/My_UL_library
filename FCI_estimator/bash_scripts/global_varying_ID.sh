#!/bin/bash


echo "Running global estimator on datasets D, G and H, for varying intrinsic dimension..."

cd ../src/

for ((id = 2; id <= 1024; id *= 2))
do
    py make_dataset.py -N 700 -D 1024 -d $id

    py fit.py --mode "g" --results_path "../results/global_varying_ID/${id}.json"
done

cd ../bash_scripts/
