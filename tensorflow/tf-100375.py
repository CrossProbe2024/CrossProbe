import tensorflow as tf
image = tf.random.uniform(shape=[3, 64, 48, 1])
boxes = tf.constant([[2.5351325e-39, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
       [3.4028235e+38, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
       [3.4028235e+38, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
       [9.2400819e-38, 3.4028235e+38, 3.4028235e+38, 3.4028235e+38],
       [1.0386443e-37, 3.4028235e+38, 5.4163033e-38, 8.3468926e-38]], dtype=tf.float32)

box_ind = tf.constant([2, 1, 1, 2, 2], dtype=tf.int32)
crop_size = tf.constant([24, 32], dtype=tf.int32)
method = "bilinear"
extrapolation_value = 0.5
output = tf.raw_ops.CropAndResize(image=image, boxes=boxes, box_ind=box_ind, crop_size=crop_size, method=method, extrapolation_value=extrapolation_value)