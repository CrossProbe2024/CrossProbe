import torch
import torch.nn as nn

a = torch.tensor([1.0000000597], requires_grad=True)
b = torch.tensor([1.0])

loss=nn.functional.binary_cross_entropy(a,b)
print(loss)
