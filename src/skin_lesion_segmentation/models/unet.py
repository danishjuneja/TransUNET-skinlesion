"""Convolutional U-Net baseline."""

from __future__ import annotations

import tensorflow as tf


def _conv_block(inputs, filters: int):
    x = inputs
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
    return x


def build_unet(image_size: int = 256, base_filters: int = 64) -> tf.keras.Model:
    """Build a four-stage U-Net for one-channel binary segmentation."""

    inputs = tf.keras.layers.Input((image_size, image_size, 3), name="image")
    skips = []
    x = inputs
    for multiplier in (1, 2, 4, 8):
        x = _conv_block(x, base_filters * multiplier)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)

    x = _conv_block(x, base_filters * 16)
    for skip, multiplier in zip(reversed(skips), (8, 4, 2, 1)):
        filters = base_filters * multiplier
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = _conv_block(x, filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x)
    return tf.keras.Model(inputs, outputs, name="UNet")

