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
    index: int,
):
    derivative_row = derivative.unsqueeze(0)
    print(index)
    print(len(matrix_chain))
    if len(matrix_chain) <= index:
        matrix_chain.append(derivative_row)
    else:
        matrix_chain[index] = torch.cat((matrix_chain[index], derivative_row), dim=0)


seed = torch.ones_like(l[0])
seed_vector = torch.ones_like(x)


def traverse_graph(function, recursion_depth, curent_seed):
    recursion_depth += 1
    next_functions = function.next_functions
    for _, function in enumerate(next_functions):
        if function[0] is None:
            continue
        derivatives = function[0](curent_seed)
        if isinstance(derivatives, tuple):
            for index, derivative in enumerate(derivatives):
                if derivative is None:
                    continue
                if index > 1:
                    raise Exception("More indices than expected")

                add_to_matrix_chain(derivative, recursion_depth)
                traverse_graph(function[0], recursion_depth, curent_seed)

        else:
            add_to_matrix_chain(derivatives, recursion_depth)
            traverse_graph(function[0], recursion_depth, curent_seed)


def construct_graph():
    for i in range(len(l)):
        recursion_depth = 0
        grad_fn = l[i].grad_fn
        derivatives = l[i].grad_fn(seed)
        if isinstance(derivatives, tuple):
            for index, derivative in enumerate(derivatives):
                if derivative is None:
                    continue
                if index > 1:
                    raise Exception("More indices than expected")

                add_to_matrix_chain(derivative, recursion_depth)
                traverse_graph(grad_fn, recursion_depth, curent_seed=derivative)
        else:
            add_to_matrix_chain(derivatives, recursion_depth)
            traverse_graph(grad_fn, recursion_depth, curent_seed=derivatives)

    print(len(matrix_chain))
    for matrix in matrix_chain:
        print(matrix)


construct_graph()
