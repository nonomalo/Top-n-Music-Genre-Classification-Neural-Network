"""Processes .npy or .npz files associated with datasets to be used as input
to a neural network model.
"""

from typing import Sequence, Tuple

import numpy as np

from constants import CHANNELS, FEATURES, SAMPLES, TIME


def create_dataset(
    inputs_filepath: str,
    labels_filepath: str
) -> Tuple[np.array, np.array]:
    """Loads .npy or .npz files associated with input data and labels
    and performs preprocessing on input data.

    :param inputs_filepath: path to .npy or .npz file associated with inputs
    :param labels_filepath: path to .npy or .npz file associated with labels
    :return: NumPy arrays representing inputs and labels, respectively
    """
    inputs = np.load(inputs_filepath)
    inputs = preprocess_inputs(inputs)
    labels = np.load(labels_filepath)
    return inputs, labels


def preprocess_inputs(inputs: Sequence[int]) -> np.array:
    """Rearranges input by swapping order of features and time dimensions
    and adding an innermost dimension representing channels.

    :param inputs: data of the shape (samples, features, time)
    :return: data of the shape (samples, time, features, channels)
    """
    inputs = np.transpose(inputs, (SAMPLES, FEATURES, TIME))
    inputs = np.expand_dims(inputs, CHANNELS)
    return inputs
