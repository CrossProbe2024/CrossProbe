import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)  # Equivalent of tf.random.set_seed(0)

# Define the Conv1D layer
conv = nn.Conv1d(in_channels=1, out_channels=65537, kernel_size=3, padding='same') # Conversion: tf.keras.layers.Conv1D -> torch.nn.Conv1d

# Batch, Length, Channel
x = torch.ones([1, 3, 1])  # Conversion: tf.ones -> torch.ones

# Move to CPU
with torch.device('cpu'):
    y_cpu = conv(x)

# Move to GPU
with torch.device('cuda'):
    y_gpu = conv(x)

print(x)
print("y_cpu:", y_cpu.detach().numpy())  # Conversion: .numpy() is equivalent for accessing the underlying numpy array
print("y_gpu:", y_gpu.detach().numpy())  # Conversion: .numpy() is equivalent for accessing the underlying numpy array