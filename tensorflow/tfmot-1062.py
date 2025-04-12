from __future__ import annotations

from typing import Callable, Optional

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers


SKIP_LAYER = [
    "resize",
    "Resize",
    "reshape",
    "Reshape",
    "concat",
    "Concat" "ExpandDims",
    "Repeats",
    "Shape",
    "strided_slice",
    "Tile",
]


def quantize_model(
    model: tf.keras.Model,
    annotate: Optional[Callable] = None,
    quantize_scope: Optional[dict[str, tf.keras.layers.Layer]] = None,
) -> tf.keras.Model:
    quantize_scope = {} if quantize_scope is None else quantize_scope

    def annotate(layer):
        if any([name in layer.name for name in SKIP_LAYER]):
            return layer
        else:
            return tfmot.quantization.keras.quantize_annotate_layer(layer)

    anno_model = tf.keras.models.clone_model(model, clone_function=annotate)
    with tfmot.quantization.keras.quantize_scope(quantize_scope):
        model = tfmot.quantization.keras.quantize_apply(anno_model)

    return model


def channel_shuffle(tensor: tf.Tensor, groups: int = 2) -> tf.Tensor:
    """Channel shuffle operation."""
    _, height, width, num_channels = tensor.shape.as_list()
    assert num_channels % groups == 0

    tensor = tf.reshape(tensor, [-1, height, width, groups, num_channels // groups])
    tensor = tf.transpose(tensor, [0, 1, 2, 4, 3])
    tensor = tf.identity(tensor, name="channel_shuffle")

    tensor = tf.reshape(tensor, [-1, height, width, num_channels])
    return tensor


def simple_nn(img_input: tf.Tensor) -> tf.Tensor:
    latent = layers.Conv2D(32, 1, padding="same", use_bias=False, name="conv1")(img_input)
    latent = layers.BatchNormalization(name="bn1")(latent)
    latent = layers.ReLU(name="relu1")(latent)

    latent = layers.DepthwiseConv2D(3, 1, padding="same", name="conv2")(img_input)
    latent = layers.BatchNormalization(name="bn2")(latent)

    latent = layers.Conv2D(32, 1, padding="same", use_bias=False, name="conv3")(img_input)
    latent = layers.BatchNormalization(name="bn3")(latent)
    latent = layers.ReLU(name="relu3")(latent)

    return latent


def split_like_nn(img_input: tf.Tensor) -> tf.Tensor:
    latent = layers.Conv2D(64, 1, padding="same", use_bias=False, name="conv0")(img_input)
    latent = layers.BatchNormalization(name="bn0")(latent)
    latent = layers.ReLU(name="relu0")(latent)

    latent_0, latent_1 = tf.split(latent, 2, axis=-1)
    latent_0 = simple_nn(latent_0)
    latent = tf.concat([latent_0, latent_1], axis=-1)

    latent = channel_shuffle(latent)

    return latent


if __name__ == "__main__":
    img_input = tf.keras.Input((128, 128, 1), dtype=tf.float32, name="img")

    outputs = split_like_nn(img_input)

    model = tf.keras.Model(inputs=img_input, outputs=outputs, name="PoseNetV2")
    model.summary()

    model_qat = quantize_model(model)
    model_qat.summary()
