import torch

a = torch.randn(101, requires_grad=True)
b = torch.randn(101, 101)
c = torch.matmul(b, a)
d = torch.matmul(b, c)

for i in range(len(d)):
    dd = d[i].grad_fn(torch.tensor(1.0))
    dc = d[i].grad_fn.next_functions[0][0](dd)
    da = d[i].grad_fn.next_functions[0][0].next_functions[1][0](dc[1])[1] / dc[1]
    print(da)
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
