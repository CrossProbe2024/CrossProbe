import torch
import torch.nn as nn

# Define batch normalization layers for CPU and MPS devices
bn_cpu = nn.BatchNorm2d(100, affine=False, device='cpu')
bn_mps = nn.BatchNorm2d(100, affine=False, device='mps')

# Create random input tensors and move them to the respective devices
x_cpu = torch.randn(100, 100, 35, 45).to('cpu')
x_mps = x_cpu.to('mps')

# Perform batch normalization on the full tensors
output_cpu = bn_cpu(x_cpu)
output_mps = bn_mps(x_mps)

# Perform batch normalization on a slice of the tensors (offset)
output_offset_cpu = bn_cpu(x_cpu[5:])
output_offset_mps = bn_mps(x_mps[5:])

# Compare the outputs.  Move the MPS output to CPU for comparison.
# Conversion comment: No conversion needed, comparing tensors on CPU.
print(f"{torch.sum(abs(output_cpu - output_mps.cpu()) > 1e-5) = }")
print(f"{torch.sum(abs(output_offset_cpu - output_offset_mps.cpu()) > 1e-5) = }")