import torch
import torch.nn as nn

bn_cpu = nn.BatchNorm2d(100, affine=False, device='cpu')
bn_mps = nn.BatchNorm2d(100, affine=False, device='mps')

x_cpu = torch.randn(100, 100, 35, 45).to('cpu')
x_mps = x_cpu.to('mps')

output_cpu = bn_cpu(x_cpu)
output_mps = bn_mps(x_mps)

output_offset_cpu = bn_cpu(x_cpu[5:])
output_offset_mps = bn_mps(x_mps[5:])

print(f"{torch.sum(abs(output_cpu - output_mps.cpu()) > 1e-5) = }")
print(f"{torch.sum(abs(output_offset_cpu - output_offset_mps.cpu()) > 1e-5) = }")
