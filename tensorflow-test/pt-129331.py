import torch

# The following code block mirrors the functionality of torch.onnx.export, 
# which is already PyTorch code.  Therefore, no translation is required.
# The original code exports a PyTorch model to ONNX format.
# Since the task is PyTorch to PyTorch, the code remains the same.
# The code exports the `net` model, given the input `input_x`, to an ONNX file named "mfnet_fp32.onnx".
# `export_params=True` includes the model's parameters in the ONNX file.
# `opset_version=12` specifies the ONNX opset version to use.
# `do_constant_folding=True` performs constant folding optimization.
# `input_names` and `output_names` specify the names of the input and output tensors.
# `dynamic_axes` specifies the dynamic dimensions for the input and output tensors, allowing for variable batch sizes.
torch.onnx.export(
   net,
   input_x,
   "mfnet_fp32.onnx",
   export_params=True,
   opset_version=12,
   do_constant_folding=True,
   input_names=["input"],
   output_names=["output"],
   dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)