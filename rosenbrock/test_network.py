import os
import pickle

import numpy as np
import torch

from rosenbrock.data_loaders.coordinate_dataset_loader import CoordinateDataset
from rosenbrock.networks.sparse_traingular_network import SpareTraingularNetwork
from rosenbrock.scaling import min_max_denormalise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_size = int(os.getenv("LAYER_SIZE"))
batch_size = int(os.getenv("BATCH_SIZE"))
test_dataset_path = os.getenv("DATASET_PATH")

testing_dataset = CoordinateDataset(test_dataset_path)

test_loader = torch.utils.data.DataLoader(
    dataset=testing_dataset, batch_size=batch_size, shuffle=False
)

model = SpareTraingularNetwork(layer_size).to(device)
model.double()
checkpoint = torch.load(
    "./rosenbrock/saved_weights/rosenbrock_trained_model_weights_sparse.pth"
)

model.load_state_dict(checkpoint)
model.eval()
with open(
    "./rosenbrock/saved_params/rosenbrock_sampled_data_normalisation_params.pkl",
    "rb",
) as f:
    denormalisation_params = pickle.load(f)

min, max = denormalisation_params["min"], denormalisation_params["max"]


with torch.no_grad():
    for i, (input_values, output_values) in enumerate(test_loader):
        input_values = input_values.to(device)
        outputs = min_max_denormalise(model(input_values), min, max)

        print("predicted value: ", outputs)
        print("exact value: ", output_values)

        diff = np.sqrt((outputs - np.array(output_values.detach().numpy())) ** 2)

        print("distance between: ", diff)
