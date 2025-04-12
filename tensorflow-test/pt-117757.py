import torch

def fn(x, y):
    # Original function: element-wise multiplication of x by the absolute value of y
    return x * abs(y)

arg = torch.ones(4, device="cuda") * 4
# Compile the function using torch.compile with fullgraph=True, backend="inductor", and dynamic=True
opt_fn = torch.compile(fullgraph=True, backend="inductor", dynamic=True)(fn)
print(opt_fn(arg, -2) == fn(arg, -2))