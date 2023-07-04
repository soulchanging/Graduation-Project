import torch
a = torch.tensor([1,2])
b = a.repeat(3,1)
c = b.repeat(5,1,1)
print(c.size())