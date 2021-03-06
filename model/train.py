"""Trains a neural network model on a dataset for music genre prediction.

Usage: python3 train.py [-h] [-l LOAD] [-s SAVE] inputs labels
"""

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf

from build import build_model
from dataset import CHANNELS
from dataset import create_dataset
from dataset import FEATURES
from dataset import load_mappings
from dataset import TIME

TRAINING_FRACTION = 0.9
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.001
CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                     restore_best_weights=True)
]


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    :return: command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--load",
        help="path to .tf or .h5 file to load preconfigured model"
    )
    parser.add_argument(
        "-s", "--save",
        help="path to .tf or .h5 file to save model"
    )
    parser.add_argument(
        "inputs",
        help="path to .npy or .npz file associated with inputs"
    )
    parser.add_argument(
        "labels",
        help="path to .npy or .npz file associated with labels"
    )

    args = parser.parse_args()
    return args


def configure_model(
    inputs: Sequence[int],
    mappings: Sequence[str]
) -> tf.keras.Model:
    """Builds and compiles a new TensorFlow model.

    :param inputs: array associated with inputs
    :param mappings: array associated with mappings
    :return: built and compiled model
    """
    input_shape = (inputs.shape[TIME], inputs.shape[FEATURES],
                   inputs.shape[CHANNELS])
    num_outputs = len(mappings)

    model = build_model(input_shape, num_outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


def display_loss(history: tf.keras.callbacks.History) -> None:
    """Displays loss metrics.

    :param history: history object returned by model after training
    :return: None
    """
    metrics = history.history

    plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation loss"])
    plt.show()


def display_accuracy(history: tf.keras.callbacks.History) -> None:
    """Displays accuracy metrics.

    :param history: history object returned by model after training
    :return: None
    """
    metrics = history.history

    plt.plot(history.epoch, metrics["accuracy"], metrics["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.show()


def test_model(
    model: tf.keras.Model,
    inputs: Sequence[int],
    labels: Sequence[int]
) -> Sequence[int]:
    """Uses trained model to make predictions on test data. Accuracy across
    entire test dataset is displayed.

    :param model: trained model
    :param inputs: array associated with test inputs
    :param labels: array associated with test labels
    :return: array associated with model predictions
    """
    predictions = np.argmax(model.predict(inputs), axis=1)
    correct = sum(predictions == labels)
    accuracy = correct / len(labels)

    print(f"Test set accuracy: {accuracy:.0%}")
    return predictions


def display_confusion_matrix(
    predictions: Sequence[int],
    labels: Sequence[int],
    mappings: Sequence[int]
) -> None:
    """Displays a confusion matrix.

    :param predictions: array associated with model predictions
    :param labels: array associated with test labels
    :param mappings: array associated with mappings
    :return: None
    """
    matrix = tf.math.confusion_matrix(labels, predictions,
                                      num_classes=len(mappings))

    plt.figure()
    sns.heatmap(
        matrix, xticklabels=mappings, yticklabels=mappings,
        annot=True, fmt="g"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    plt.show()


def main() -> None:
    """Trains a neural network model on a dataset for music genre prediction.

    :return: None
    """
    args = get_arguments()
    inputs, labels = create_dataset(args.inputs, args.labels)
    mappings = load_mappings()

    if args.load:
        model = tf.keras.models.load_model(args.load)
    else:
        model = configure_model(inputs, mappings)
    model.summary()

    training_inputs, remainder_inputs, training_labels, remainder_labels = (
        train_test_split(inputs, labels,
                         train_size=TRAINING_FRACTION, shuffle=True)
    )
    validation_inputs, test_inputs, validation_labels, test_labels = (
        train_test_split(remainder_inputs, remainder_labels,
                         test_size=0.5, shuffle=True)
    )

    history = model.fit(
        training_inputs, training_labels,
        batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=CALLBACKS,
        validation_data=(validation_inputs, validation_labels)
    )
    display_loss(history)
    display_accuracy(history)

    predictions = test_model(model, test_inputs, test_labels)
    display_confusion_matrix(predictions, test_labels, mappings)

    if args.save:
        model.save(args.save)


if __name__ == "__main__":
    main()
