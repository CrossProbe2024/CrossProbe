import torch
import torch.nn.functional as F

# Ensure inputs are within [0, 1]
a = torch.tensor([1.0000000597], dtype=torch.float32)  # Convert tf.Variable to torch.tensor
b = torch.tensor([1.0], dtype=torch.float32)  # Convert tf.constant to torch.tensor

# Using the binary_crossentropy function
# tf.keras.losses.binary_crossentropy is equivalent to torch.nn.functional.binary_cross_entropy
loss = F.binary_cross_entropy(a, b)  # Conversion comment: Using PyTorch equivalent
print(loss)