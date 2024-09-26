import csv
import os

import numpy as np

from rosenbrock.generate_data.function import rosenbrock_nd_gradient

layer_size = int(os.getenv("LAYER_SIZE"))
number_of_datapoints = int(os.getenv("NUMBER_OF_DATAPOINTS"))
dataset_path = os.getenv("DATASET_PATH")


def generate_data(no_of_data_points: int, min: float = -2.0, max: float = -2.0):
    # x = np.linspace(-2.0, 2.0, no_of_data_points_sqrt)
    # y = np.linspace(-1.0, 3.0, no_of_data_points_sqrt)

    # for i in range(no_of_data_points_sqrt):
    #     for j in range(no_of_data_points_sqrt):
    #         input = np.array([x[i], y[j]])
    #         output = rosenbrock_nd_gradient(input)

    #         data = [input, output]

    #         _save_data_to_csv(data)

    for i in range(no_of_data_points):
        input = np.random.uniform(min, max, size=(layer_size))
        output = rosenbrock_nd_gradient(input)

        data = [input, output]

        _save_data_to_csv(data)

        if (i + 1) % 1000 == 0:
            print(f"run [{i + 1}/{no_of_data_points}] done")


def _save_data_to_csv(data: list[np.ndarray]):
    with open("dataset_path", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        csv_writer.writerow(data[1])


generate_data(number_of_datapoints)
