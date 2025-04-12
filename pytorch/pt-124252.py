import torch
x=torch.arange(3, dtype=torch.float16,device='mps')
print(torch.nn.functional.binary_cross_entropy(x, x))
