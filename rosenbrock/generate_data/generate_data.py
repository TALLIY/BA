import csv

import numpy as np

from rosenbrock.generate_data.constants import LOCAL_DATASET_PATH
from rosenbrock.generate_data.function import rosenbrock_nd_gradient


def generate_data(no_of_data_points: int):
    # x = np.linspace(-2.0, 2.0, no_of_data_points_sqrt)
    # y = np.linspace(-1.0, 3.0, no_of_data_points_sqrt)

    # for i in range(no_of_data_points_sqrt):
    #     for j in range(no_of_data_points_sqrt):
    #         input = np.array([x[i], y[j]])
    #         output = rosenbrock_nd_gradient(input)

    #         data = [input, output]

    #         _save_data_to_csv(data)

    for i in range(no_of_data_points):
        input = np.random.uniform(-2.0, 2.0, size=(20))
        output = rosenbrock_nd_gradient(input)

        data = [input, output]

        _save_data_to_csv(data)

        if (i + 1) % 1000 == 0:
            print(f"run [{i + 1}/{no_of_data_points}] done")


def _save_data_to_csv(data: list[np.ndarray]):
    with open(
        f"{LOCAL_DATASET_PATH}/rosenbrock_training_data.csv", "a", newline=""
    ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(data[0])
        csv_writer.writerow(data[1])


generate_data(10000000)
