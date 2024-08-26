import torch

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


x = torch.tensor([8.0, 8.0], requires_grad=True)
y = torch.tensor([[4.0, 4.0], [4.0, 4.0]])

w = torch.matmul(y, x)
v = torch.matmul(y, w)
p = v**2
l = torch.matmul(y, p)

num_operations = 5
counter = 0
matrix_chain = []


def increment():
    global counter
    counter += 1
    counter = counter % len(matrix_chain)


def add_to_matrix_chain(
    derivative: torch.Tensor,
):
    derivative_row = derivative.unsqueeze(0)
    if len(matrix_chain) < num_operations:
        matrix_chain.append(derivative_row)
    else:
        matrix_chain[counter] = torch.cat(
            (matrix_chain[counter], derivative_row), dim=0
        )
        increment()


seed = torch.ones_like(l[0])
seed_vector = torch.ones_like(x)

for i in range(len(l)):
    dl = l[i].grad_fn(seed)
    add_to_matrix_chain(dl)

    dp = l[i].grad_fn.next_functions[0][0](dl)
    add_to_matrix_chain(dp[1])

    dv = l[i].grad_fn.next_functions[0][0].next_functions[1][0](dl)
    add_to_matrix_chain(dv)

    dw = l[i].grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0](dl)
    add_to_matrix_chain(dw[1])

    dx = (
        l[i]
        .grad_fn.next_functions[0][0]
        .next_functions[1][0]
        .next_functions[0][0]
        .next_functions[1][0](dl)
    )
    add_to_matrix_chain(dx[1])


for matrix in matrix_chain:
    print(matrix)
