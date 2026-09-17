"""Hybrid CNN-Transformer U-Net.

Adapted from awsaf49/TransUNet-tf under the MIT License. See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math
from typing import Sequence

import tensorflow as tf


class AddPositionEmbeddings(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="position_embedding",
            shape=(1, input_shape[1], input_shape[2]),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.embedding, inputs.dtype)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, heads: int, mlp_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        hidden_size = int(input_shape[-1])
        if hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=hidden_size // self.heads,
            dropout=self.dropout_rate,
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.mlp_dim, activation=tf.nn.gelu),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(hidden_size),
                tf.keras.layers.Dropout(self.dropout_rate),
            ]
        )

    def call(self, inputs, training=None):
        normalized = self.norm1(inputs)
        attended = self.attention(normalized, normalized, training=training)
        x = inputs + self.dropout(attended, training=training)
        return x + self.mlp(self.norm2(x), training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "heads": self.heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
        }


def _conv_relu(inputs, filters: int):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def _decoder_block(inputs, skip, filters: int):
    x = tf.keras.layers.UpSampling2D(2, interpolation="bilinear")(inputs)
    if skip is not None:
        x = tf.keras.layers.Concatenate()([x, skip])
    x = _conv_relu(x, filters)
    return _conv_relu(x, filters)


def build_transunet(
    image_size: int = 256,
    hidden_size: int = 768,
    transformer_layers: int = 12,
    heads: int = 12,
    mlp_dim: int = 3072,
    dropout: float = 0.1,
    decoder_channels: Sequence[int] = (256, 128, 64, 16),
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """Build TransUNet with a ResNet50V2 hybrid encoder."""

    if image_size % 32:
        raise ValueError("image_size must be divisible by 32")
    if hidden_size % heads:
        raise ValueError("hidden_size must be divisible by heads")
    if len(decoder_channels) != 4:
        raise ValueError("decoder_channels must contain four stages")

    inputs = tf.keras.layers.Input((image_size, image_size, 3), name="image")
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_tensor=inputs,
    )
    backbone.trainable = not freeze_backbone
    encoded = backbone.get_layer("conv4_block6_preact_relu").output
    skips = [
        backbone.get_layer("conv3_block4_preact_relu").output,
        backbone.get_layer("conv2_block3_preact_relu").output,
        backbone.get_layer("conv1_conv").output,
    ]

    tokens = tf.keras.layers.Conv2D(hidden_size, 1, name="token_projection")(encoded)
    grid_size = int(tokens.shape[1])
    tokens = tf.keras.layers.Reshape((grid_size * grid_size, hidden_size))(tokens)
    tokens = AddPositionEmbeddings(name="position_embeddings")(tokens)
    tokens = tf.keras.layers.Dropout(dropout)(tokens)
    for index in range(transformer_layers):
        tokens = TransformerBlock(heads, mlp_dim, dropout, name=f"transformer_{index}")(
            tokens
        )
    tokens = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(tokens)

    token_count = int(tokens.shape[1])
    side = int(math.sqrt(token_count))
    if side * side != token_count:
        raise ValueError("Transformer token grid is not square")
    x = tf.keras.layers.Reshape((side, side, hidden_size))(tokens)
    x = _conv_relu(x, 512)
    for index, filters in enumerate(decoder_channels):
        skip = skips[index] if index < len(skips) else None
        x = _decoder_block(x, skip, filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x)
    return tf.keras.Model(inputs, outputs, name="TransUNet")

