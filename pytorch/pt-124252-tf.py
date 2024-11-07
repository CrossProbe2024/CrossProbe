import tensorflow as tf

# Create a tensor with values from 0 to 2, with dtype float32 (float16 often isn't supported for many operations in TensorFlow)
x = tf.range(3, dtype=tf.float32)

# TensorFlow does not support binary cross entropy directly between two hard tensors without proper probabilistic values,
# We need to apply a sigmoid function to simulate probabilities (since BCE expects probabilities).
x_sigmoid = tf.sigmoid(x)

# Compute binary cross entropy between the same tensors
loss = tf.keras.losses.binary_crossentropy(x_sigmoid, x_sigmoid)

print("Binary Cross Entropy:", loss.numpy())
