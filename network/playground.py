import torch

a = torch.randn(2, requires_grad=True)
b = torch.tensor([[8.0, 0.0], [0.0, 2.0]])
c = torch.matmul(b, a)
d = torch.matmul(b, c)
e = torch.matmul(b, d)


# Define a hook function to monitor the gradient
def hook_fn(grad):
    print(f"Gradient in hook: {grad}")


# Register the hook on the tensor x
hook_handle3 = d.register_hook(hook_fn)
hook_handle2 = c.register_hook(hook_fn)
hook_handle = a.register_hook(hook_fn)


# Loop through each element of y and call .backward() individually
for i in range(len(e)):
    # Zero out the gradients for a clean backward pass
    a.grad = None

    # Call backward on the scalar y[i]
    e[i].backward(retain_graph=True)

    # Print the gradient for x with respect to y[i]
    # print(f"Gradient of y[{i}] with respect to x: {a.grad}")

# Clean up the hook
hook_handle.remove()
hook_handle2.remove()
hook_handle3.remove()


# print(d[i].grad_fn)
# print(unbound_d[i].grad_fn.next_functions[0][0].next_functions[1][0])


# x = torch.randn(4, 4, requires_grad=True)
# y = torch.randn(4)
# z = torch.matmul(x, y)
# w = x + y + z
# l = torch.pow(w, 2)


# dl = torch.tensor(1.0)


# back_sum = l.grad_fn
# dz = back_sum(dl)
# back_mul = back_sum.next_functions[0][0]
# print(dz)
# print("this is z")


# dx, dy = back_mul(dz)
# print(dx / dz)
# print(dy / dz)


# back_x = back_mul.next_functions[0][0]
# back_x(dx)
# back_y = back_mul.next_functions[1][0]
# back_y(dy)

# print("this is x")

# print([item for item in dir(z) if item not in dir(x)])

# print(x.grad)
# print(y.grad)
