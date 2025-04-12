import tensorflow as tf
import numpy as np

# Set the random seed
tf.random.set_seed(0)

# Define a 1D convolution layer
conv = tf.keras.layers.Conv1D(filters=65537, kernel_size=3, padding='same')

# Create the input tensor
x = tf.ones([1, 1, 3])  # Note the shape difference: TensorFlow expects (batch_size, length, channels)

# Perform the convolution on CPU
with tf.device('/CPU:0'):
    y_cpu = conv(x)

# Perform the convolution on GPU (if available)
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        # Recreate the convolution layer to reset weights (needed because TensorFlow doesn't automatically share weights across devices like PyTorch)
        conv_gpu = tf.keras.layers.Conv1D(filters=65537, kernel_size=3, padding='same')
        # conv_gpu.set_weights(conv.get_weights())  # Copy weights from CPU layer to GPU layer
        y_gpu = conv_gpu(x)
else:
    print("WARN: no GPU available")
    y_gpu = y_cpu  # Fallback to CPU if no GPU is available

print("y_cpu:", y_cpu.numpy())
print("y_gpu:", y_gpu.numpy())
print("Equal:", np.allclose(y_cpu.numpy(), y_gpu.numpy()))
