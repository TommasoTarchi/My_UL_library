#!/bin/bash


echo "Running global estimator on datasets D, G and H, for varying dataset size..."

cd ../src/

for ((size = 50; size <= 1000; size += 50))
do
    py make_dataset.py -N $size -D 1000 -d 400

    py fit.py --mode "g" --results_path "../results/global_varying_size/${size}.json"
done

cd ../bash_scripts/
