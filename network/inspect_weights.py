import os

import numpy as np
import torch
from btcs import BTCS
from dotenv import load_dotenv
from networks import dense_network, sparse_network

load_dotenv()
train_dense_network = True

if os.getenv("TRAIN_DENSE_NETWORK") == "0":
    train_dense_network = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if train_dense_network:
    model = dense_network(99).to(device)
else:
    model = sparse_network(99).to(device)
model.float()

checkpoint = torch.load(
    "../shared/weights/model_weights_for_layer_size_99_for_dense_False.pth"
)
model.load_state_dict(checkpoint)


# # Print the weights of the model
# for name, param in model.named_parameters():
#     print(f"{name}: {param}")


sim = BTCS(length=1, delta_x=0.005, alpha=0.01, delta_t=0.01, min_T=0, max_T=1000)

suff_f = sim.infer_final_step_value(
    model=model,
    state_dict_path="../shared/weights/model_weights_for_layer_size_99_for_dense_False.pth",
    denormalisation_params_path="../shared/parameters/min_max_scaling_params_for_layer_size_99_for_dense_False.pkl",
    compare=False,
)

M = np.identity(99).dot(suff_f.numpy())

state_dict_items = list(model.state_dict().items())
reversed_state_dict_items = reversed(state_dict_items)
for i in range(0, len(state_dict_items) - 1, 2):
    k1, v1 = next(reversed_state_dict_items)
    k2, v2 = next(reversed_state_dict_items)
    print(f"Index: {i}, Key: {k1}, {k2}")
    M = M.dot(v1.detach().numpy() * v2.detach().numpy())

print(M)
