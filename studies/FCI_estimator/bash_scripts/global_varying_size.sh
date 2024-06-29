#!/bin/bash


echo "Running global estimator on datasets D, G and H, for varying dataset size..."
echo "-- intrinsic dimension is set to 400 for all datasets"
echo "-- embedding dimension is set to 1000 for all datasets"

cd ../src/

for ((size = 50; size <= 1000; size += 50))
do
    python3 make_datasets.py -N $size -D 1000 -d 400 --noise_std_dev 0.08

    python3 fit.py --mode "g" --results_path "../results/global_varying_size/${size}.json"
done

cd ../bash_scripts/
