"""Evaluates a neural network model on a dataset for music genre prediction.

Usage: python3 evaluate.py [-h] [-d DISPLAY] filepath [filepath ...] model
"""

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load_data, load_mappings, SAMPLES
from train import get_inputs_and_labels


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    :return: command line arguments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath", nargs="+",
        help="path to JSON file associated with dataset"
    )
    parser.add_argument(
        "model",
        help="path to savedModel or H5 file to load preconfigured model"
    )
    parser.add_argument(
        "-d", "--display", type=int,
        help="number of results to display"
    )

    args = parser.parse_args()
    return args


def verify_arguments(arguments: argparse.Namespace) -> None:
    """Verifies command line arguments.

    :param arguments: command line arguments object
    :raises ValueError: when number of results to display is not positive
    :return: None
    """
    if arguments.display and arguments.display < 1:
        raise ValueError("Number of results to display must be at least one")


def evaluate_model(
    model: tf.keras.Model,
    inputs: Sequence[int],
    mappings: Sequence[int],
    num_display: int
) -> None:
    """Evaluates model on a dataset and displays predictions for
    a specified number of inputs.

    :param model: preconfigured model
    :param inputs: array associated with inputs
    :param mappings: array associated with mappings
    :param num_display: number of results to display
    :return: None
    """
    for i in range(num_display):
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        plt.bar(mappings, tf.nn.softmax(prediction[0]))
        plt.title("Prediction")
        plt.ylabel("Probability")
        plt.show()


def main() -> None:
    """Evaluates a neural network model on a dataset for music genre prediction.

    :return: None
    """
    args = get_arguments()
    verify_arguments(args)

    model = tf.keras.models.load_model(args.model)
    dataset = load_data(args.filepath)
    mappings = load_mappings(args.filepath[0])

    inputs, _ = get_inputs_and_labels(dataset)
    evaluate_model(model, inputs, mappings, args.display or 1)


if __name__ == "__main__":
    main()
