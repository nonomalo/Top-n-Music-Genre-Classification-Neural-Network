"""Builds a neural network model."""

from typing import Sequence

import tensorflow as tf


def build_model(
    input_shape: Sequence[int],
    outputs: int
) -> tf.keras.Model:
    """Builds a 2D convolutional neural network model composed of three
    convolutional blocks followed by one fully connected block.

    :param input_shape: feature data dimensionality
    :param outputs: number of all possible categories to predict
    :return: model with layers added
    """
    model = tf.keras.models.Sequential()

    # Convolutional block 1
    model.add(tf.keras.layers.Input(input_shape))
    model.add(tf.keras.layers.Resizing(128, 128))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convolutional block 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convolutional block 3
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(2, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Fully connected block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(outputs))

    return model
