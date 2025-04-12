import torch
 import torch.nn.functional as F

 # Create a tensor with values from 0 to 2, with dtype float32
 x = torch.arange(3, dtype=torch.float32)

 # TensorFlow's sigmoid function is equivalent to torch.sigmoid
 x_sigmoid = torch.sigmoid(x)

 # Compute binary cross entropy between the same tensors
 # PyTorch's binary cross entropy function is equivalent to TensorFlow's.
 # Note: PyTorch expects input to be probabilities, similar to TensorFlow.
 loss = F.binary_cross_entropy(x_sigmoid, x_sigmoid)

 print("Binary Cross Entropy:", loss.item())