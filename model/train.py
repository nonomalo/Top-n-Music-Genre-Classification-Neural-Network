"""Trains and evaluates a neural network model on a dataset
for music genre prediction.

usage: train.py [-h] [-s SAVE] filepath [filepath ...]
"""

import argparse
import random
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from build import build_model
from dataset import load_data, load_mappings, SAMPLES

BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.0002


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments.

    :return: command line arguments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", nargs="+",
                        help="path to JSON file associated with dataset")
    parser.add_argument("-s", "--save",
                        help="path to SavedModel or H5 file to save model")

    args = parser.parse_args()
    return args


def get_element_shape(dataset: tf.data.Dataset) -> tf.TensorShape:
    """Determines dimensionality of a single element from the dataset.

    :param dataset: TensorFlow dataset
    :return: dimensionality of a single element
    """
    for element, _ in dataset.take(1):
        shape = element.shape
    return shape


def split_dataset(
    dataset: tf.data.Dataset,
    split_fraction: float
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits TensorFlow dataset into two.

    :param dataset: TensorFlow dataset to split
    :param split_fraction: fraction of original dataset to allocate
    :return: two TensorFlow datasets
    """
    split_percent = round(split_fraction * 100)

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data:
                                   f % 100 > split_percent)
    remainder_dataset = dataset.filter(lambda f, data:
                                       f % 100 <= split_percent)

    # Remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    remainder_dataset = remainder_dataset.map(lambda f, data: data)

    return train_dataset, remainder_dataset


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


def evaluate_model(
    model: tf.keras.Model,
    mappings: Sequence[int],
    dataset: tf.data.Dataset
) -> None:
    """Uses trained model to make predictions on test data. Accuracy
    on entire test dataset is displayed and confidence values are
    displayed on a subset of the test dataset.

    :param model: trained model
    :param mappings: array associated with mappings data
    :param dataset: TensorFlow dataset of test inputs and labels
    :return: None
    """
    inputs, labels = get_inputs_and_labels(dataset)

    predicted = np.argmax(model.predict(inputs), axis=1)
    correct = sum(predicted == labels)
    accuracy = correct / len(labels)
    print(f"Test set accuracy: {accuracy:.0%}")

    for _ in range(10):
        i = random.randrange(len(inputs) - 1)
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        plt.bar(mappings, tf.nn.softmax(prediction[0]))
        plt.title(f"Predictions for '{mappings[labels[i]]}'")
        plt.ylabel("Probability")
        plt.show()


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
    """Trains and evaluates a neural network model on a dataset
    for music genre prediction.

    :return: None
    """
    args = get_arguments()

    dataset = load_data(args.filepath)
    mappings = load_mappings(args.filepath[0])

    input_shape = get_element_shape(dataset)
    model = build_model(input_shape, len(mappings))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    model.summary()

    train_dataset, remainder = split_dataset(dataset, 0.1)
    validation_dataset, test_dataset = split_dataset(remainder, 0.5)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    history = model.fit(train_dataset, epochs=EPOCHS,
                        validation_data=validation_dataset,
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1,
                                                                   patience=2)
                        )
    display_training_metrics(history)
    evaluate_model(model, mappings, test_dataset)

    if args.save:
        model.save(args.save)


if __name__ == "__main__":
    main()
