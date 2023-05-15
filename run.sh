#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/logistic_regression.py
python3 src/neural_net.py

# deactivate 
deactivate