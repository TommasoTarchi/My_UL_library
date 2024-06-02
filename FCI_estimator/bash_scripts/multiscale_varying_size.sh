#!/bin/bash


echo "Running multiscale estimator on datasets C and B, for varying dataset size..."
echo "-- intrinsic dimension of C is set to 400, while that of B is set to 15 (3 blobs)"
echo "-- embedding dimension is set to 800 for both C and B"

cd ../src/

for ((size = 50; size <= 1000; size += 50))
do
    python3 make_datasets.py -N $size -D 800 -d 400 -n 3

    python3 fit.py --mode "m" --results_path "../results/multiscale_varying_size/${size}.json"
done

cd ../bash_scripts/
