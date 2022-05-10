"""Builds, trains, and evaluates a 2D convolutional neural network
model on a dataset for music genre prediction.

Usage: python3 model.py json-filepath [json-filepath ...]
"""

import json
import random
import sys
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
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

# Tunable parameters
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.0002


def load_data(paths: Sequence[str]) -> tf.data.Dataset:
    """Loads JSON files to retrieve input data and labels,
    performs preprocessing, and creates a TensorFlow dataset.

    :param paths: file paths
    :return: TensorFlow dataset with preprocessed data
    """
    try:
        for i, path in enumerate(paths):
            with open(path, "r") as file:
                data = json.load(file)

            inputs = np.array(data[FEATURE_DATA])
            labels = np.array(data[LABEL_DATA])
            inputs = preprocess_data(inputs)

            if i == 0:
                dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            else:
                temp = tf.data.Dataset.from_tensor_slices((inputs, labels))
                dataset = dataset.concatenate(temp)
    except FileNotFoundError:
        sys.exit(f"'{path}' is an invalid file path")
    except KeyError:
        sys.exit(f"'{path}' has an incompatible JSON structure")

    return dataset


def preprocess_data(inputs: Sequence[int]) -> np.array:
    """Normalizes features data and rearranges input by swapping order
    of features and time dimensions and adding an innermost dimension
    representing channels.

    :param inputs: data of the shape (samples, features, time)
    :return: data of the shape (samples, time, features, channels)
    """
    scalar = StandardScaler()
    num_samples, num_features, time_steps = inputs.shape

    # Normalization
    inputs = np.reshape(inputs, newshape=(-1, num_features))
    inputs = scalar.fit_transform(inputs)

    # Rearrangement
    inputs = np.reshape(inputs,
                        newshape=(num_samples, time_steps, num_features)
                        )
    inputs = np.expand_dims(inputs, CHANNELS)

    return inputs


def load_mappings(path: str) -> np.array:
    """Loads JSON file to retrieve mappings data.

    :param path: file path
    :return: array associated with mappings
    """
    with open(path, "r") as file:
        data = json.load(file)

    mappings = np.array(data[MAPPING_DATA])
    return mappings


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
                                   f % 100 > split_percent
                                   )
    remainder_dataset = dataset.filter(lambda f, data:
                                       f % 100 <= split_percent
                                       )

    # Remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    remainder_dataset = remainder_dataset.map(lambda f, data: data)

    return train_dataset, remainder_dataset


def get_element_shape(dataset: tf.data.Dataset) -> tf.TensorShape:
    """Determines dimensionality of a single element from the dataset.

    :param dataset: TensorFlow dataset
    :return: dimensionality of a single element
    """
    for element, _ in dataset.take(1):
        shape = element.shape
    return shape


def build_model(
    input_shape: Sequence[int],
    num_labels: int
) -> tf.keras.Model:
    """Builds a 2D convolutional neural network model composed of three
    convolutional blocks followed by one fully connected block.

    :param input_shape: feature data dimensionality
    :param num_labels: number of all possible output categories
    :return: model with layers added
    """
    model = tf.keras.models.Sequential()

    # Convolutional block 1
    model.add(tf.keras.layers.Input(input_shape))
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
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_labels))

    return model


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
    """Builds, trains, and evaluates a 2D convolutional neural network
    model on a dataset for music genre prediction.

    :return: None
    """
    dataset = load_data(sys.argv[1:])
    mappings = load_mappings(sys.argv[1])

    input_shape = get_element_shape(dataset)
    model = build_model(input_shape, len(mappings))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
        )
    model.summary()

    train_dataset, remainder_dataset = split_dataset(dataset, 0.1)
    validation_dataset, test_dataset = split_dataset(remainder_dataset, 0.5)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    history = model.fit(train_dataset, epochs=EPOCHS,
                        validation_data=validation_dataset,
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1,
                                                                   patience=3
                                                                   )
                        )
    display_training_metrics(history)
    evaluate_model(model, mappings, test_dataset)


if __name__ == "__main__":
    main()
