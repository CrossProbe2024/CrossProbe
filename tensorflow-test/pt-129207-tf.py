import torch
 import torch.nn as nn
 import numpy as np

 # Set the random seed
 torch.manual_seed(0)

 # Define a 1D convolution layer
 # TensorFlow's Conv1D has input shape (batch_size, length, channels),
 # PyTorch's Conv1d has input shape (batch_size, channels, length).
 # We need to transpose the input accordingly.
 conv = nn.Conv1d(in_channels=3, out_channels=65537, kernel_size=3, padding='same')

 # Create the input tensor
 # TensorFlow expects (batch_size, length, channels)
 # PyTorch expects (batch_size, channels, length)
 x = torch.ones([1, 3, 3]) # Note the shape difference

 # Perform the convolution on CPU
 with torch.device('cpu'):
   y_cpu = conv(x)

 # Perform the convolution on GPU (if available)
 if torch.cuda.is_available():
   with torch.device('cuda'):
     # Recreate the convolution layer to reset weights (needed because TensorFlow doesn't automatically share weights across devices like PyTorch)
     conv_gpu = nn.Conv1d(in_channels=3, out_channels=65537, kernel_size=3, padding='same')
     # In PyTorch, we don't need to manually copy weights, as the same layer can be used on different devices.
     y_gpu = conv_gpu(x)
 else:
   print("WARN: no GPU available")
   y_gpu = y_cpu # Fallback to CPU if no GPU is available

 print("y_cpu:", y_cpu.numpy())
 print("y_gpu:", y_gpu.numpy())
 print("Equal:", np.allclose(y_cpu.numpy(), y_gpu.numpy()))