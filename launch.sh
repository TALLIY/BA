#!/bin/bash


export PYTHONPATH="$(pwd)"
export TRAIN_DENSE_NETWORK=0



pip install -r requirements.txt

python $(pwd)/rosenbrock/generate_data/generate_data.py
python $(pwd)/rosenbrock//train_network.py


