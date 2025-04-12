import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

conv = tf.keras.layers.Conv1D(filters=65537, kernel_size=3, padding='same')

x = tf.ones([1, 1, 3])

with tf.device('/CPU:0'):
    y_cpu = conv(x)

with tf.device('/GPU:0'):
    y_gpu = conv(x)

print("y_cpu:", y_cpu.numpy())
print("y_gpu:", y_gpu.numpy())
print("Equal:", np.allclose(y_cpu.numpy(), y_gpu.numpy()))
