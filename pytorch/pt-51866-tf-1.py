import tensorflow as tf

with tf.device('/GPU:0'):
    # Ensure inputs are within [0, 1]
    a = tf.Variable([1.0000000597], dtype=tf.float32)
    b = tf.constant([1.0], dtype=tf.float32)

    # Using the binary_crossentropy function
    loss = tf.keras.losses.binary_crossentropy(a, b)

print(loss)
