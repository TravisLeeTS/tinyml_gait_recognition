from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tiny_ds_cnn(input_shape: tuple[int, int], num_classes: int) -> keras.Model:
    """Compact 1D DS-CNN candidate for TensorFlow Lite Micro experiments."""
    inputs = keras.Input(shape=input_shape, name="imu_window")
    x = layers.Conv1D(12, kernel_size=3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = layers.SeparableConv1D(24, kernel_size=5, padding="same", use_bias=False, name="ds_conv_1")(x)
    x = layers.BatchNormalization(name="ds_bn_1")(x)
    x = layers.ReLU(name="ds_relu_1")(x)
    x = layers.AveragePooling1D(pool_size=2, name="avg_pool_1")(x)

    x = layers.SeparableConv1D(32, kernel_size=5, padding="same", use_bias=False, name="ds_conv_2")(x)
    x = layers.BatchNormalization(name="ds_bn_2")(x)
    x = layers.ReLU(name="ds_relu_2")(x)
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="activity")(x)
    return keras.Model(inputs, outputs, name="tiny_ds_cnn")


def count_trainable_parameters(model: keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
