"""Evaluates a neural network model on a dataset for music genre prediction.

Usage: python3 evaluate.py [-h] [-l LABELS] [-d DISPLAY] inputs model
"""

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from dataset import load_mappings, preprocess_inputs
from dataset import SAMPLES
from train import test_model


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    :return: command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--labels",
        help="path to .npy or .npz file associated with labels"
    )
    parser.add_argument(
        "-d", "--display", default=1, type=int,
        help="number of predictions to display"
    )
    parser.add_argument(
        "inputs",
        help="path to .npy or .npz file associated with inputs"
    )
    parser.add_argument(
        "model",
        help="path to .tf or .h5 file to load preconfigured model"
    )

    args = parser.parse_args()
    return args


def evaluate_model(
    model: tf.keras.Model,
    inputs: Sequence[int],
    mappings: Sequence[int],
    display: int = 1,
    labels: Sequence[int] = None
) -> None:
    """Evaluates model on a dataset and displays predictions for
    a specified number of inputs.

    :param model: preconfigured model
    :param inputs: array associated with input
    :param mappings: array associated with mappings
    :param display: number of predictions to display, optional
    :param labels: array associated with labels, optional
    :return: None
    """
    for i in range(display):
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        title = "Prediction"
        if labels is not None:
            label = mappings[labels[i]]
            title += f" for {label}"

        plt.title(title)
        plt.bar(mappings, tf.nn.softmax(prediction[0]))
        plt.xlabel("Genres")
        plt.ylabel("Probability")
        plt.show()


def main() -> None:
    """Evaluates a neural network model on a dataset for music genre prediction.

    :return: None
    """
    args = get_arguments()

    model = tf.keras.models.load_model(args.model)
    inputs = np.load(args.inputs)
    inputs = preprocess_inputs(inputs)
    mappings = load_mappings()

    if args.labels:
        labels = np.load(args.labels)
        inputs, labels = shuffle(inputs, labels)
        evaluate_model(model, inputs, mappings, args.display, labels)
        test_model(model, inputs, labels, mappings)
    else:
        inputs = shuffle(inputs)
        evaluate_model(model, inputs, mappings, args.display)


if __name__ == "__main__":
    main()
