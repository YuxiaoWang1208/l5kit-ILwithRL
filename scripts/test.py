import torch

# a = torch.arange(2).unsqueeze(-1)
# b = torch.arange(2,4).unsqueeze(-1)
# print(a, b)

# c = torch.cat([a, b], 0)
# print(c)

# cost_dist = torch.tensor([[0, 0.5, 1], [1.5, 2, 2.5]])
# print(cost_dist)
# assignment = cost_dist.argmin(dim=-1)
# print(assignment)
# loss_dist = cost_dist[[0,1], assignment]
# print(loss_dist)

# cost_dist = torch.tensor([[[0, 0.5, 1], [7, 0.5, 1]], [[1.5, 2, 2.5], [8, 2, 2.5]]])
# print(cost_dist)
# assignment = cost_dist.argmin(dim=-1)
# print(assignment)
# loss_dist = cost_dist[[0, 1], [0, 1], assignment[0]]
# print(loss_dist)

# a = torch.zeros([1,1,1,2])
# print(a.shape)
# b = a.repeat(1,1,2,1)
# print(b.shape)
# print(b)
# c = a.repeat(2,1,1,1)
# print(c)

a = torch.tensor([[[2,4], [6,8]], [[4,4], [12,8]]])
print(a.shape)
b = torch.tensor([[2,4], [6,8]])
print(b.shape)
c = a / b
print(c)

s = "global_head_1"
if "global_head" in s:
    print("yes")
