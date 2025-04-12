import torch

# Create a zero tensor of integer type
x_i32 = torch.zeros([3, 4], dtype=torch.int32) 

print(x_i32)

# Create a zero tensor of float type
x_f32 = torch.zeros([3, 4], dtype=torch.float32)

print(x_f32)