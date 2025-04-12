import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)  # Equivalent to tf.random.set_seed(0)

# Define the 1D Convolution layer
conv = nn.Conv1d(in_channels=3, out_channels=65537, kernel_size=3, padding='same')

# Create a sample input tensor
x = torch.ones(1, 3, 1)  # Changed shape to (batch_size, channels, seq_len)

# Perform computation on CPU
with torch.device('cpu'):
    y_cpu = conv(x)

# Perform computation on GPU
with torch.device('cuda'):  # Assuming CUDA is available
    y_gpu = conv(x)

# Print the results
print("y_cpu:", y_cpu.detach().numpy())
print("y_gpu:", y_gpu.detach().numpy())
print("Equal:", np.allclose(y_cpu.detach().numpy(), y_gpu.detach().numpy()))