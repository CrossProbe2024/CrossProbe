import tensorflow as tf

features = tf.constant([], shape=[3,0], dtype=tf.float64)
labels = tf.constant([], shape=[0], dtype=tf.float64)
tf.raw_ops.SoftmaxCrossEntropyWithLogits(features=features, labels=labels)
