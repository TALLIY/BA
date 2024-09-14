# import os
# import pickle

# import numpy as np
# from dotenv import load_dotenv
# from rosenbrock.generate_data.function import rosenbrock_gradient
# from rosenbrock.networks.dense_network import DenseNetwork
# load_dotenv()
# train_dense_network = True
# if os.getenv("TRAIN_DENSE_NETWORK") == "0":
#     train_dense_network = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DenseNetwork(2).to(device)
# model.float()
# checkpoint = torch.load(
#     "/Users/talli/BA/rosenbrock/saved_weights/rosenbrock_trained_model_weights_dense.pth"
# )
# model.load_state_dict(checkpoint)
# model.eval()
# with open(
#     "/Users/talli/BA/rosenbrock/saved_params/rosenbrock_sampled_data_dense.pkl", "rb"
# ) as f:
#     denormalisation_params = pickle.load(f)
# min, max = denormalisation_params["min"], denormalisation_params["max"]
# x = np.random.uniform(-2.0, 2.0)
# y = np.random.uniform(-1.0, 3.0)
# input_np = np.array([x, y])
# input = torch.tensor(input_np.astype(np.float32), requires_grad=True)
# surr_f = ((model(input) * (max - min)) + min).detach().numpy()
# rosenbrock_grad = rosenbrock_gradient(input_np)
# print(surr_f)
# print(rosenbrock_grad)

import torch
import torch.nn as nn

layer_size = 3


parameters_list = [nn.Parameter(torch.rand(layer_size - i)) for i in range(layer_size)]


x = torch.tensor([3.0, 2.0, 1.0])
y = torch.zeros(3)

dim = len(parameters_list)

# for i in range(layer_size):
#     len_vector = len(parameters_list[i])
#     for j in range(len_vector):
#         y[i] += parameters_list[i][j] * x[layer_size - len_vector + j]

# print(y)


for i in reversed(range(len(parameters_list))):
    parameters_list[i] = torch.cat(
        (
            parameters_list[i],
            torch.zeros(i),
        ),
        dim=0,
    )

matrix = torch.stack([component.squeeze(0) for component in parameters_list], dim=0)


print(matrix)
print(type(matrix))
print(matrix.shape)

for param in parameters_list:
    print(param)
