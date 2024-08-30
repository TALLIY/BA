import csv

import numpy as np

from rosenbrock.generate_data.constants import LOCAL_DATASET_PATH
from rosenbrock.generate_data.function import rosenbrock_gradient


def generate_data(no_of_data_points: int):
    for i in range(no_of_data_points):
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-3, 6)
        input = np.array([x, y])

        grad = rosenbrock_gradient(input)

        data = [input, grad]

        _save_data_to_csv(data)

        if (i + 1) % 100 == 0:
            print(f"run [{i + 1}/{no_of_data_points}] done")


def _save_data_to_csv(data: list[np.ndarray]):
    with open(
        f"{LOCAL_DATASET_PATH}/rosenbrock_testing_data.csv", "a", newline=""
    ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        csv_writer.writerow(data[1])


generate_data(20000)
