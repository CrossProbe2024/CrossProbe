import torch

def _randn(size: list[int], device: torch.device):
    generator = torch.Generator(device=device)
    return torch.randn(size, device=device, generator=generator)
	
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):
        val = _randn(x.shape, torch.device("cpu"))
        return val

m = MyModule()
torch.onnx.export(MyModule(), torch.randn(1, 2), 'test.onnx')
