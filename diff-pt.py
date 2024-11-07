import torch
import torch.nn as nn
torch.manual_seed(0)

conv = nn.Conv1d(1, 65537, 3, padding=1)

# Batch, Channel, Length

x = torch.ones([1, 1, 3])
y_cpu = conv.to("cpu")(x.to("cpu"))
y_mps = conv.to("mps")(x.to("mps"))

print(x)
print(y_cpu.detach().numpy())
print(y_mps.to("cpu").detach().numpy())
