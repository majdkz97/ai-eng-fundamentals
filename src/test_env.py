import torch

a = torch.empty(4, 1, 6)
b = torch.empty(3, 1)
c = a + b

print(c.shape)