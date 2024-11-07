import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

conv = tf.keras.layers.Conv1D(filters=65537, kernel_size=3, padding='same')

# Batch, Length, Channel

x = tf.ones([1, 3, 1])

with tf.device('/CPU:0'):
    y_cpu = conv(x)

with tf.device('/GPU:0'):
    y_gpu = conv(x)

print(x)
print("y_cpu:", y_cpu.numpy())
print("y_gpu:", y_gpu.numpy())
