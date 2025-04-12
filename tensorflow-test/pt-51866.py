import torch
import torch.nn as nn

# Define tensors
a = torch.tensor([1.0000000597], requires_grad=True)  # requires_grad=True indicates that gradients should be computed for this tensor
b = torch.tensor([1.0])

# Calculate binary cross entropy loss
loss = nn.functional.binary_cross_entropy(a, b)  # This function calculates the binary cross entropy loss between the input tensors.
print(loss)