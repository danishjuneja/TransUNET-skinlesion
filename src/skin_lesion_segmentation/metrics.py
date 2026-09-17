"""Differentiable binary-segmentation metrics."""

from __future__ import annotations

import tensorflow as tf


SMOOTH = 1e-7


@tf.keras.utils.register_keras_serializable(package="skin_lesion_segmentation")
def dice_coefficient(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + SMOOTH) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH
    )


@tf.keras.utils.register_keras_serializable(package="skin_lesion_segmentation")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="skin_lesion_segmentation")
def soft_iou(y_true, y_pred):
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)

