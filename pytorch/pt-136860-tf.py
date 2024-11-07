import tensorflow as tf
import numpy as np

def _randn(shape):
    return tf.random.normal(shape)

class MyModule(tf.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    @tf.function
    def __call__(self, x):
        val = _randn(x.shape)
        return val

# Create an instance of MyModule
m = MyModule()

# Example input tensor
input_tensor = tf.random.normal([1, 2])

# Exporting to ONNX
import onnx
import tf2onnx

# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_function(
    m.__call__,  # function to export
    input_signature=[tf.TensorSpec([1, 2], tf.float32)],
    output_path='test.onnx'
)

