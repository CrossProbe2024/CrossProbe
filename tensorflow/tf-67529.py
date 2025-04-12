import tensorflow as tf

mode = "REFLECT"
strides = [1, 1, 1, 1]
padding = "VALID"
resize_align_corners = False
input = tf.constant(0, shape=[1,2,3,2], dtype=tf.float16)
size = tf.constant([65534,65534], shape=[2], dtype=tf.int32)
paddings = tf.constant(0, shape=[4,2], dtype=tf.int32)
filter = tf.constant(0, shape=[1,2,2,2], dtype=tf.float16)
tf.raw_ops.FusedResizeAndPadConv2D(input=input, size=size, paddings=paddings, filter=filter, mode=mode, strides=strides, padding=padding, resize_align_corners=resize_align_corners)
