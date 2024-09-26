import os
import pickle

import torch
import torch.nn as nn
from dotenv import load_dotenv

from rosenbrock.data_loaders.coordinate_dataset_loader import CoordinateDataset
from rosenbrock.networks.dense_network import DenseNetwork
from rosenbrock.networks.sparse_traingular_network import SpareTraingularNetwork
from rosenbrock.scaling import (
    min_max_normalise,
)

load_dotenv()
train_dense_network = True

if os.getenv("TRAIN_DENSE_NETWORK") == "0":
    train_dense_network = False

print("Training Sparse Network: ", not train_dense_network)

torch.autograd.set_detect_anomaly(True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
layer_size = int(os.getenv("LAYER_SIZE"))
num_epochs = int(os.getenv("NUM_EPOCHS"))
batch_size = int(os.getenv("BATCH_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
training_dataset_path = os.getenv("DATASET_PATH")


# training and test datasets
train_dataset = CoordinateDataset(training_dataset_path)

# Normalization
params_file_path = (
    "rosenbrock/saved_params/rosenbrock_sampled_data_normalisation_params.pkl"
)

if os.path.exists(params_file_path):
    with open(params_file_path, "rb") as f:
        min_max_scaling_params = pickle.load(f)
else:
    train_loader_norm = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
    )

    min_value = float("inf")
    max_value = float("-inf")

    for _, (input_values, output_values) in enumerate(train_loader_norm):
        current_min = torch.min(output_values)
        current_max = torch.max(output_values)

        min_value = min(min_value, current_min)
        max_value = max(max_value, current_max)

    min_max_scaling_params = {"min": min_value, "max": max_value}
    print(min_max_scaling_params)

    with open(params_file_path, "wb") as f:
        pickle.dump(min_max_scaling_params, f)


# Data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)


# model
if train_dense_network:
    model = DenseNetwork(layer_size).to(device)
else:
    model = SpareTraingularNetwork(layer_size).to(device)

model.double()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (input_values, output_values) in enumerate(train_loader):
        # resize
        input_values = input_values.to(device)
        output_values = min_max_normalise(
            output_values.to(device),
            min_max_scaling_params["min"],
            min_max_scaling_params["max"],
        )
        # Forward pass
        outputs = model(input_values.double())
        loss = criterion(outputs, output_values)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.8f}"
            )

    scheduler.step()


torch.save(
    model.state_dict(),
    f'rosenbrock/saved_weights/rosenbrock_trained_model_weights_{"dense" if train_dense_network else "sparse"}.pth',
)
