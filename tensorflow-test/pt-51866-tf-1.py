import torch
 import torch.nn.functional as F

 # Ensure inputs are within [0, 1]
 a = torch.tensor([1.0000000597], dtype=torch.float32, requires_grad=True) # Converted tf.Variable to torch.tensor with requires_grad=True for gradient calculation
 b = torch.tensor([1.0], dtype=torch.float32) # Converted tf.constant to torch.tensor

 # Using the binary_cross_entropy function
 # tf.keras.losses.binary_crossentropy is equivalent to torch.nn.functional.binary_cross_entropy
 loss = F.binary_cross_entropy(a, b) # Converted tf.keras.losses.binary_crossentropy to F.binary_cross_entropy

 print(loss)