import os
import pickle

import torch
import torch.nn as nn
from coordinate_dataset_loader import coordinate_dataset
from dotenv import load_dotenv
from networks import dense_network, sparse_network

load_dotenv()
train_dense_network = False

if os.getenv("TRAIN_DENSE_NETWORK") == "0":
    train_dense_network = False

print(train_dense_network)

torch.autograd.set_detect_anomaly(True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
layer_size = 99
num_epochs = 25
batch_size = 100
learning_rate = 0.0001

# training and test datasets
train_dataset = coordinate_dataset("../shared/datasets/training.csv")
test_dataset = coordinate_dataset("../shared/datasets/test.csv")

# normalisation
train_loader_norm = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=len(train_dataset), shuffle=False
)

min_value = float("inf")
max_value = float("-inf")
for input_values, output_values in enumerate(train_loader_norm):
    concat_tensor = torch.cat(output_values, dim=0)
    min_value = torch.min(concat_tensor)
    max_value = torch.max(concat_tensor)

min_max_scaling_params = {"min": min_value, "max": max_value}
with open(
    f"../shared/parameters/min_max_scaling_params_for_layer_size_{layer_size}_for_dense_{str(train_dense_network)}.pkl",
    "wb",
) as f:
    pickle.dump(min_max_scaling_params, f)


# Data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# model
if train_dense_network:
    model = dense_network(layer_size).to(device)
else:
    model = sparse_network(layer_size).to(device)
model.float()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (input_values, output_values) in enumerate(train_loader):
        # resize
        input_values = input_values.to(device)
        output_values = (output_values.to(device) - min_value) / (max_value - min_value)
        # Forward pass
        outputs = model(input_values.float())
        loss = criterion(outputs, output_values)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.6f}"
            )

# Test the model
with torch.no_grad():
    total_r2 = []
    for input_values, output_values in test_loader:
        input_values = input_values.to(device)
        outputs = (model(input_values) * (max_value - min_value)) + min_value

        # calculate the mean and the variance
        mean_input_values = torch.mean(input_values, dim=0)
        squared_diff = (input_values - mean_input_values) ** 2
        variance = torch.mean(squared_diff, dim=0)

        # calculate the mean squared error
        squared_diff = (input_values - outputs) ** 2
        mse = torch.mean(squared_diff, dim=0)

        variance += 1e-6
        r2 = torch.ones(layer_size) - (mse / variance)

        total_r2.append(r2)

    total_r2_stack = torch.stack(total_r2)
    mean_total_r2 = torch.mean(total_r2_stack, dim=0)

    mean_r2 = torch.mean(mean_total_r2).item()
    max_r2 = torch.max(mean_total_r2).item()
    min_r2 = torch.min(mean_total_r2).item()

    print(
        f"mean coefficient of determination of the network on the test data: {mean_r2}"
    )
    print(f"max coefficient of determination of the network on the test data: {max_r2}")
    print(f"min coefficient of determination of the network on the test data: {min_r2}")

    test_results = {"mean_cod": mean_r2, "min_cod": min_r2, "max_cod": max_r2}
    with open(
        f"../shared/parameters/test_results_for_layer_size_{layer_size}_for_dense_{str(train_dense_network)}.pkl",
        "wb",
    ) as f:
        pickle.dump(test_results, f)


torch.save(
    model.state_dict(),
    f"../shared/weights/model_weights_for_layer_size_{layer_size}_for_dense_{str(train_dense_network)}.pth",
)
