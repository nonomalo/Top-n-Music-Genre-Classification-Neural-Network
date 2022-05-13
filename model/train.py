"""Trains a neural network model on a dataset for music genre prediction.

Usage: python3 train.py [-h] [-l LOAD] [-s SAVE] inputs labels
"""

import argparse
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from build import build_model
from dataset import create_dataset, load_mappings

TRAIN_FRACTION = 0.8  # Remainder is split evenly between validation and test
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.0005


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    :return: command line arguments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--load",
        help="path to SavedModel or H5 file to load preconfigured model"
    )
    parser.add_argument(
        "-s", "--save",
        help="path to SavedModel or H5 file to save model"
    )
    parser.add_argument(
        "inputs",
        help="path to NumPy file associated with inputs"
    )
    parser.add_argument(
        "labels",
        help="path to NumPy file associated with labels"
    )

    args = parser.parse_args()
    return args


def configure_model(
    dataset: tf.data.Dataset,
    mappings: Sequence[str]
) -> tf.keras.Model:
    """Builds and compiles a new TensorFlow model.

    :param dataset: TensorFlow dataset
    :param mappings: array associated with mappings
    :return: built and compiled model
    """
    input_shape = get_element_shape(dataset)
    outputs = len(mappings)

    model = build_model(input_shape, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


def get_element_shape(dataset: tf.data.Dataset) -> tf.TensorShape:
    """Determines dimensionality of a single element from the dataset.

    :param dataset: TensorFlow dataset
    :return: dimensionality of a single element
    """
    for element, _ in dataset.take(1):
        shape = element.shape
    return shape


def create_training_validation_test_datasets(
    dataset: tf.data.Dataset,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Splits dataset into training, validation, and test datasets
    based on TRAIN_FRACTION. After the training dataset is created,
    the remainder is split evenly between the validation and test datasets.

    :param dataset: TensorFlow dataset
    :return: datasets for training, validation, and test, respectively
    """
    remainder_fraction = 1.0 - TRAIN_FRACTION

    training_ds, remainder_ds = split_dataset(dataset, remainder_fraction)
    validation_ds, test_ds = split_dataset(remainder_ds, 0.5)

    return training_ds, validation_ds, test_ds


def split_dataset(
    dataset: tf.data.Dataset,
    split_fraction: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits TensorFlow dataset into two.

    :param dataset: TensorFlow dataset to split
    :param split_fraction: fraction of original dataset to be allocated
    :return: reduced original and newly allocated datasets, respectively
    """
    split_percent = round(split_fraction * 100)

    dataset = dataset.enumerate()
    original_ds = dataset.filter(lambda f, data:
                                 f % 100 > split_percent)
    remainder_ds = dataset.filter(lambda f, data:
                                  f % 100 <= split_percent)

    # Remove enumeration
    original_ds = original_ds.map(lambda f, data: data)
    remainder_ds = remainder_ds.map(lambda f, data: data)

    return original_ds, remainder_ds


def display_training_metrics(history: tf.keras.callbacks.History) -> None:
    """Displays loss metrics associated with model during training.

    :param history: history object returned by model after training
    :return: None
    """
    metrics = history.history

    plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
    plt.title("Training Loss Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation loss"])
    plt.show()


def test_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> None:
    """Uses trained model to make predictions on test data. Accuracy
    on entire test dataset is displayed.

    :param model: trained model
    :param dataset: TensorFlow dataset with test inputs and labels
    :return: None
    """
    inputs, labels = get_inputs_and_labels(dataset)

    predicted = np.argmax(model.predict(inputs), axis=1)
    correct = sum(predicted == labels)
    accuracy = correct / len(labels)
    print(f"Test set accuracy: {accuracy:.0%}")


def get_inputs_and_labels(
    dataset: tf.data.Dataset
) -> Tuple[np.array, np.array]:
    """Retrieves inputs and labels from a TensorFlow dataset.

    :param dataset: TensorFlow dataset
    :return: arrays associated with inputs and labels, respectively
    """
    inputs = []
    labels = []

    for input, label in dataset:
        inputs.append(input)
        labels.append(label)

    inputs = np.array(inputs)
    labels = np.array(labels)
    return inputs, labels


def main() -> None:
    """Trains a neural network model on a dataset for music genre prediction.

    :return: None
    """
    args = get_arguments()

    dataset = create_dataset(args.inputs, args.labels)
    dataset = dataset.shuffle(dataset.__len__())

    if args.load:
        model = tf.keras.models.load_model(args.load)
    else:
        mappings = load_mappings()
        model = configure_model(dataset, mappings)
    model.summary()

    training_ds, validation_ds, test_ds \
        = create_training_validation_test_datasets(dataset)

    training_ds = training_ds.batch(BATCH_SIZE)
    validation_ds = validation_ds.batch(BATCH_SIZE)

    history = model.fit(training_ds, epochs=EPOCHS,
                        validation_data=validation_ds,
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1,
                                                                   patience=3)
                        )
    display_training_metrics(history)
    test_model(model, test_ds)

    if args.save:
        model.save(args.save)


if __name__ == "__main__":
    main()
