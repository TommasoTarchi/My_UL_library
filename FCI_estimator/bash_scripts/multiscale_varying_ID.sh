#!/bin/bash

echo "Building datasets..."

cd ../src/
python3 make_dataset.py  # PARAMETERS

echo "Running estimator..."

###

echo "Cleaning up..."

###
