import torch
import torch.nn as nn
from computational_graph_builder import ComputationalGrapBuilder

# a = torch.randn(2, requires_grad=True)
# b = torch.tensor([[8.0, 0.0], [0.0, 2.0]])
# c = torch.matmul(b, a)
# d = torch.matmul(b, c)
# e = torch.matmul(b, d)


# # Define a hook function to monitor the gradient
# def hook_fn(grad):
#     print(f"Gradient in hook: {grad}")


# # Register the hook on the tensor x
# hook_handle3 = d.register_hook(hook_fn)
# hook_handle2 = c.register_hook(hook_fn)
# hook_handle = a.register_hook(hook_fn)


# # Loop through each element of y and call .backward() individually
# for i in range(len(e)):
#     # Zero out the gradients for a clean backward pass
#     a.grad = None

#     # Call backward on the scalar y[i]
#     e[i].backward(retain_graph=True)

#     # Print the gradient for x with respect to y[i]
#     # print(f"Gradient of y[{i}] with respect to x: {a.grad}")

# # Clean up the hook
# hook_handle.remove()
# hook_handle2.remove()
# hook_handle3.remove()


# print(d[i].grad_fn)
# print(unbound_d[i].grad_fn.next_functions[0][0].next_functions[1][0])


# x = torch.tensor(8.0, requires_grad=True)
# y = torch.tensor(4.0)

# w = x * y

# v = w * y

# l = v * y

# dl = torch.tensor(1.0)

# dv = l.grad_fn(dl)
# print(dv[0])

# dw = l.grad_fn.next_functions[0][0](torch.tensor(1.0))
# print(dw[0])

# dx = l.grad_fn.next_functions[0][0].next_functions[0][0](dw[0])
# print(dx[0] / dw[0])


# x = torch.tensor([8.0, 8.0], requires_grad=True)
# y = torch.tensor([[4.0, 4.0], [4.0, 4.0]])

# w = nn.Linear(2, 2)(x)
# v = nn.Linear(2, 2)(w)
# p = v**2
# l = nn.Linear(2, 2)(p)

# cgb = ComputationalGrapBuilder()
# chain = cgb.construct_graph(l)

# for matrix in chain:
#     print(matrix)

s = torch.tensor([8.0, 8.0], requires_grad=True)
layer = nn.Linear(2, 2)
layer2 = nn.Linear(2, 2)
layer3 = nn.Linear(2, 2)

with torch.no_grad():
    layer.weight = nn.Parameter(torch.tensor([[4.0, 4.0], [4.0, 4.0]]))
    layer.bias = nn.Parameter(torch.tensor([0.0, 0.0]))
    layer2.weight = nn.Parameter(torch.tensor([[4.0, 4.0], [4.0, 4.0]]))
    layer2.bias = nn.Parameter(torch.tensor([0.0, 0.0]))
    layer3.weight = nn.Parameter(torch.tensor([[4.0, 4.0], [4.0, 4.0]]))
    layer3.bias = nn.Parameter(torch.tensor([0.0, 0.0]))

X = layer(s)
Y = layer2(X)
Y_0 = torch.pow(Y, 2)
Z = layer3(Y_0)

cgb = ComputationalGrapBuilder()
chain = cgb.construct_graph(Z)

for matrix in chain:
    print(matrix)


# def generate_seed(tensor):
#     if tensor.shape[0] == 1:
#         return torch.tensor([[1.0, 0.0]])
#     else:
#         return torch.tensor([1.0, 0.0]).unsqueeze(0)


# d1 = Z[0].grad_fn(torch.tensor(1.0))
# print(d1)

# d2 = Z[0].grad_fn.next_functions[0][0](generate_seed(d1))
# print(d2)


# d3 = Z[0].grad_fn.next_functions[0][0].next_functions[0][0](generate_seed(d2))
# print(d3)

# d4 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0](generate_seed(d3[1]))
# )
# print(d4)

# d5 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0](generate_seed(d4))
# )
# print(d5)

# d6 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0](generate_seed(d5))
# )
# print(d6)

# d7 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0](generate_seed(d6))
# )
# print(d7)


# d8 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0](generate_seed(d7[1]))
# )
# print(d8)


# d9 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0](generate_seed(d8))
# )
# print(d9)

# d10 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0](generate_seed(d9))
# )
# print(d10)

# d11 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0](generate_seed(d10))
# )


# d12 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0](generate_seed(d11[1]))
# )


# d13 = (
#     Z[0]
#     .grad_fn.next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions[0][0]
#     .next_functions[0][0]
#     .next_functions[1][0]
#     .next_functions
# )

# print(d13)
