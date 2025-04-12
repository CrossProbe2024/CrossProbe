import torch
import torch.distributed as dist

torch.library.define(
   "my_lib::my_function",
   "(Tensor x, __torch__.torch.classes.c10d.ProcessGroup y) -> Tensor"
)

def my_function(x: torch.Tensor, y: dist.ProcessGroup) -> torch.Tensor:
   # Define the 'my_function' operator.
   return torch.empty_like(x)

torch.library.impl("my_lib::my_function", "default", my_function)

@torch.library.impl_abstract("my_lib::my_function")
def my_function_abstract(x, y):
   # Define the abstract implementation of 'my_function'.
   return torch.empty_like(x)


x = torch.randn((2, 3))
g = torch.distributed.ProcessGroup(0, 0)
boxed_group = g.boxed()

@torch.compile(backend="eager", fullgraph=True)
def f(x, y):
   # Define a function that uses the custom operator.
   return torch.ops.my_lib.my_function.default(x, y)


f(x, boxed_group)