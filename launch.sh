#!/bin/bash

#TARGET_DIR="./talli"

#mkdir -p "$TARGET_DIR"

#git clone https://github.com/TALLIY/BA.git "$TARGET_DIR"

#d "$TARGET_DIR" || exit 1 

conda activate bachelor-arbeit

export PYTHONPATH="$(pwd)"

#pip install -r requirements.txt

export TRAIN_DENSE_NETWORK=0
export LAYER_SIZE=10
export NUM_EPOCHS=100
export BATCH_SIZE=1000
export LEARNING_RATE=0.001

# mkdir -p "$(pwd)/rosenbrock/datasets"

export NUMBER_OF_DATAPOINTS=1000000
export DATASET_PATH="$(pwd)/rosenbrock/datasets/rosenbrock_training_data.csv"
python $(pwd)/rosenbrock/generate_data/generate_data.py

export NUMBER_OF_DATAPOINTS=10000
export DATASET_PATH="$(pwd)/rosenbrock/datasets/rosenbrock_testing_data.csv"
python $(pwd)/rosenbrock/generate_data/generate_data.py


python $(pwd)/rosenbrock/train_network.py

export BATCH_SIZE=1
python $(pwd)/rosenbrock/test_network.py

