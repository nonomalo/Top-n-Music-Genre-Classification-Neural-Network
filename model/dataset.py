"""Processes NumPy files associated with datasets to be used as input
to a neural network model.
"""

from typing import Sequence

import numpy as np
import tensorflow as tf

SAMPLES = 0
TIME = 1
FEATURES = 2
CHANNELS = 3


def create_dataset(
    inputs_filepath: str,
    labels_filepath: str
) -> tf.data.Dataset:
    """Loads NumPy files associated with input data and labels,
    performs preprocessing on input data, and creates a TensorFlow dataset.

    :param inputs_filepath: path to NumPy file associated with inputs
    :param labels_filepath: path to NumPy file associated with labels
    :return: TensorFlow dataset with preprocessed data
    """
    inputs = np.load(inputs_filepath)
    inputs = preprocess_inputs(inputs)
    labels = np.load(labels_filepath)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset


def preprocess_inputs(inputs: Sequence[int]) -> np.array:
    """Rearranges input by swapping order of features and time dimensions
    and adding an innermost dimension representing channels.

    :param inputs: data of the shape (samples, features, time)
    :raises ValueError: when provided incompatible input data shape
    :return: data of the shape (samples, time, features, channels)
    """
    if inputs.ndim != 3:
        raise ValueError("Input data shape must be three dimensions")
    num_samples, num_features, time_steps = inputs.shape

    inputs = np.reshape(inputs,
                        newshape=(num_samples, time_steps, num_features))
    inputs = np.expand_dims(inputs, CHANNELS)
    return inputs


def load_mappings() -> np.array:
    """Returns mappings data.

    :return: array associated with mappings
    """
    mappings = [
        "International", "Blues", "Jazz", "Classical",
        "Old-Time / Historic", "Country", "Pop", "Rock",
        "Easy Listening", "Soul-RnB", "Electronic",
        "Folk", "Spoken", "Hip-Hop", "Experimental",
        "Instrumental"
    ]
    return mappings
