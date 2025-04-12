import torch
import torch.nn as nn
torch.manual_seed(0)

# Define a 1D convolutional layer
conv = nn.Conv1d(1, 65537, 3, padding=1)

# Create a dummy input tensor
x = torch.ones([1, 1, 3])

# Perform the convolution on the CPU
y_cpu = conv.to("cpu")(x.to("cpu"))

# Perform the convolution on the MPS device
y_mps = conv.to("mps")(x.to("mps"))

# Check if the outputs are equal after moving the MPS output to the CPU
print("Equal:", torch.equal(y_cpu, y_mps.to("cpu")))

# Print the CPU output as a NumPy array
print(y_cpu.detach().numpy())

# Print the MPS output (moved to CPU) as a NumPy array
print(y_mps.to("cpu").detach().numpy())