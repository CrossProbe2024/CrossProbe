import torch
# Original code: import torch
# x=torch.arange(3, dtype=torch.float16,device='mps')
# print(torch.nn.functional.binary_cross_entropy(x, x))
x = torch.arange(3, dtype=torch.float16, device='mps')
# Conversion comment: No changes are needed for this line.
print(torch.nn.functional.binary_cross_entropy(x, x))
# Conversion comment: No changes are needed for this line.