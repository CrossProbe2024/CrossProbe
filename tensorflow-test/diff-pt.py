import torch
import torch.nn as nn
torch.manual_seed(0)

conv = nn.Conv1d(1, 65537, 3, padding=1)

# Batch, Channel, Length
x = torch.ones([1, 1, 3])
y_cpu = conv.to("cpu")(x.to("cpu")) # Move both model and input to CPU
y_mps = conv.to("mps")(x.to("mps")) # Move both model and input to MPS

print(x)
print(y_cpu.detach().numpy())
print(y_mps.to("cpu").detach().numpy()) # Move MPS tensor to CPU before converting to numpy