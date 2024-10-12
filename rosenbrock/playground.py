import os
import pickle

import numpy as np
import torch
from dotenv import load_dotenv

from network.computational_graph_builder import ComputationalGrapBuilder
from rosenbrock.generate_data.function import rosenbrock_nd_gradient
from rosenbrock.networks.sparse_traingular_network import SpareTraingularNetwork
from rosenbrock.scaling import min_max_denormalise

load_dotenv()
train_dense_network = True
if os.getenv("TRAIN_DENSE_NETWORK") == "0":
    train_dense_network = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpareTraingularNetwork(2).to(device)
model.double()
checkpoint = torch.load(
    "/Users/talli/BA/rosenbrock/saved_weights/rosenbrock_trained_model_weights_sparse_prod.pth"
)

model.load_state_dict(checkpoint)
model.eval()
with open(
    "/Users/talli/BA/rosenbrock/saved_params/rosenbrock_sampled_data_normalisation_params_2.pkl",
    "rb",
) as f:
    denormalisation_params = pickle.load(f)
min, max = denormalisation_params["min"], denormalisation_params["max"]

input_np = np.array([1.0, 0.0])
input = torch.tensor(input_np.astype(np.float64), requires_grad=True)
rosenbrock_grad = rosenbrock_nd_gradient(input_np)
sur = model(input)

f_surr = min_max_denormalise(sur, min, max)
print(sur)

cgb = ComputationalGrapBuilder()
matrix_chain = cgb.construct_graph(sur)


print(
    torch.tensor([[-0.5961, 0.0000], [0.0517, -0.4507]])
    @ torch.tensor([[-0.9373, 1.5036], [0.0000, 1.5633]])
    @ torch.tensor([[1.0000, 0.0000], [0.0000, 0.0100]])
    @ torch.tensor([[-1.4935, 0.0000], [0.4343, 2.1688]])
    @ torch.tensor([[3.7157, 0.3103], [0.0000, -1.9756]])
    @ torch.tensor([[0.0100, 0.0000], [0.0000, 1.0000]])
    @ torch.tensor([[-1.0642, 0.0000], [0.5029, -0.7959]])
    @ torch.tensor([[-2.4911, -0.4599], [0.0000, -1.3042]])
    @ torch.tensor([[0.0100, 0.0000], [0.0000, 1.0000]])
    @ torch.tensor([[-1.5181, 0.0000], [-0.9812, -0.9695]])
    @ torch.tensor([[1.5743, 0.3415], [0.0000, -0.7797]])
    @ torch.tensor([1.0, 0.0])
)

# seed = torch.tensor([1.0, 0.0])
# d0_fn = sur.grad_fn
# print(d0_fn)
# d0 = d0_fn(seed)
# print(d0)

# d1_fn = sur.grad_fn.next_functions[0][0]
# print(d1_fn)
# d1 = d1_fn(seed)
# print(d1)

# d2_fn = sur.grad_fn.next_functions[0][0].next_functions
# print(d2_fn)
# d2 = d2_fn[0][0](seed)
# print(d2)

# d3_fn = sur.grad_fn.next_functions[0][0].next_functions[0][0].next_functions
# print(d3_fn)
# d3 = d3_fn[0][0](seed.unsqueeze(0))
# print(d3)

# d4_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d4_fn)
# d4 = d4_fn[0][0](seed)
# print(d4)


# d5_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d5_fn)
# d5 = d5_fn[0][0](seed)
# print(d5)

# d7_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d7_fn)
# d7 = d7_fn[0][0](seed)
# print(d7)

# d8_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d8_fn)
# d8 = d8_fn[0][0](seed.unsqueeze(0))
# print(d8)

# d9_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d9_fn)
# d9 = d9_fn[0][0](seed)
# print(d9)


# d10_fn = (
#     sur.grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions
# )
# print(d10_fn)
# d10 = d10_fn[0][0](seed)
# print(d10)


# layer_size = 3


# parameters_list = nn.ParameterList(
#     [nn.Parameter(torch.rand(layer_size - i)) for i in range(layer_size)]
# )


# x = torch.tensor([3.0, 2.0, 1.0])
# y = torch.zeros(3)

# dim = len(parameters_list)

# # for i in range(layer_size):
# #     len_vector = len(parameters_list[i])
# #     for j in range(len_vector):
# #         y[i] += parameters_list[i][j] * x[layer_size - len_vector + j]

# # print(y)


# for i in range(len(parameters_list)):
#     parameters_list[i] = torch.cat(
#         (
#             parameters_list[i],
#             torch.zeros(len(parameters_list) - len(parameters_list[i])),
#         ),
#         dim=0,
#     )

# # matrix = torch.stack([component.squeeze(0) for component in parameters_list], dim=0)


# # print(matrix)
# # print(type(matrix))
# # print(matrix.shape)

# for param in parameters_list:
#     print(param)


# parameter_top = nn.Parameter(torch.rand([1]))
# parameter_bottom = nn.Parameter(torch.rand([2]))

# print(
#     torch.stack(
#         [
#             torch.cat(
#                 (
#                     parameter_top,
#                     torch.tensor([0]),
#                 ),
#                 dim=0,
#             ),
#             parameter_bottom,
#         ],
#         dim=0,
#     )
# )
