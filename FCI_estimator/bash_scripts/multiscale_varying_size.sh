#!/bin/bash


cd ../src/

#echo "Running multiscale estimator on datasets C and B, for varying dataset size..."
#echo "-- intrinsic dimension of C is set to 400, while that of B is set to 15 (3 blobs)"
#echo "-- embedding dimension for C is set to 800"

#for ((size = 600; size <= 2000; size += 200))
#do
#    python3 make_datasets.py -N $size -D 60 -d 30 -n 3
#
#    python3 fit.py --mode "m" --results_path "../results/multiscale_varying_size/${size}.json"
#done


echo "Running multiscale estimator on datasets C and B, with dataset size fixed to 2000..."
echo "-- intrinsic dimension of C is set to 30, while that of B is set to 15 (3 blobs)"
echo "-- embedding dimension for C is set to 60"

python3 make_datasets.py -N 2000 -D 60 -d 30 -n 3
python3 fit.py --mode "m" --results_path "../results/multiscale_varying_size/fixed_size.json"

cd ../bash_scripts/
