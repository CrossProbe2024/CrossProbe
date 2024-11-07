import torch

def fn(x, y):
    return x * abs(y)

arg = torch.ones(4, device="cuda") * 4
opt_fn = torch.compile(fullgraph=True, backend="inductor", dynamic=True)(fn)
print(opt_fn(arg, -2) == fn(arg, -2))
