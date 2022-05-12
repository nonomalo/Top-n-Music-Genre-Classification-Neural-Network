"""Processes JSON files associated with datasets to be used as input
to a neural network model.
"""

import json
import sys
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

SAMPLES = 0
TIME = 1
FEATURES = 2
CHANNELS = 3

MAPPING_DATA = "mapping"
FEATURE_DATA = "mfcc"
LABEL_DATA = "labels"


def load_data(paths: Sequence[str]) -> tf.data.Dataset:
    """Loads JSON files to retrieve input data and labels,
    performs preprocessing, and creates a TensorFlow dataset.

    :param paths: file paths
    :return: TensorFlow dataset with preprocessed data
    """
    for i, path in enumerate(paths):
        with open(path, "r") as file:
            data = json.load(file)

        try:
            inputs = np.array(data[FEATURE_DATA])
            labels = np.array(data[LABEL_DATA])
        except KeyError as e:
            sys.exit(f"KeyError: JSON structure in '{path}' missing {e}")

        inputs = preprocess_data(inputs)

        if i == 0:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        else:
            temp = tf.data.Dataset.from_tensor_slices((inputs, labels))
            dataset = dataset.concatenate(temp)
    return dataset


def preprocess_data(inputs: Sequence[int]) -> np.array:
    """Normalizes features data and rearranges input by swapping order
    of features and time dimensions and adding an innermost dimension
    representing channels.

    :param inputs: data of the shape (samples, features, time)
    :return: data of the shape (samples, time, features, channels)
    """
    scalar = StandardScaler()

    if inputs.ndim != 3:
        raise ValueError("Feature data shape must be three dimensions")
    num_samples, num_features, time_steps = inputs.shape

    # Normalization
    inputs = np.reshape(inputs, newshape=(-1, num_features))
    inputs = scalar.fit_transform(inputs)

    # Rearrangement
    inputs = np.reshape(inputs,
                        newshape=(num_samples, time_steps, num_features))
    inputs = np.expand_dims(inputs, CHANNELS)
    return inputs


def load_mappings(path: str) -> np.array:
    """Loads JSON file to retrieve mappings data.

    :param path: file path
    :return: array associated with mappings
    """
    with open(path, "r") as file:
        data = json.load(file)

    try:
        mappings = np.array(data[MAPPING_DATA])
    except KeyError as e:
        sys.exit(f"KeyError: JSON structure in '{path}' missing {e}")
    return mappings
