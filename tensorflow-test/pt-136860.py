import torch

def _randn(size: list[int], device: torch.device):
   # Create a generator for reproducibility
   generator = torch.Generator(device=device)
   # Generate a random tensor with the specified size and device
   return torch.randn(size, device=device, generator=generator)
	
class MyModule(torch.nn.Module):
   def __init__(self):
       super(MyModule, self).__init__()

   def forward(self, x):
       # Generate a random tensor with the same shape as the input
       val = _randn(x.shape, torch.device("cpu"))
       return val

# Instantiate the module
m = MyModule()
# Export the module to ONNX format.  Input is a random tensor of shape (1, 2).
torch.onnx.export(MyModule(), torch.randn(1, 2), 'test.onnx')