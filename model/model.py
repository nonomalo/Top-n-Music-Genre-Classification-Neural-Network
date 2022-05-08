"""Builds, trains, and evaluates a 2D convolutional neural network
model on a dataset for music genre prediction.

Usage: python3 model.py json-filepath ...
"""

import json
import random
import sys
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

SAMPLES = 0
TIME = 1
FEATURES = 2
CHANNELS = 3

MAPPING_DATA = "mapping"
INPUT_DATA = "mfcc"
LABEL_DATA = "labels"


def verify_arguments() -> None:
    """Performs initial checks associated with file paths before
    building and training model.

    :return: None
    """
    try:
        for arg in sys.argv[1:]:
            with open(arg, "r") as file:
                data = json.load(file)

            data[MAPPING_DATA]
            data[INPUT_DATA]
            data[LABEL_DATA]
    except IndexError:
        sys.exit(f"Usage: python3 {sys.argv[0]} json-filepath ...")
    except FileNotFoundError:
        sys.exit(f"'{arg}' is an invalid file path")
    except KeyError:
        sys.exit(f"'{arg}' has an incompatible JSON structure")


def load_data(path: str) -> Tuple[np.array, np.array, np.array]:
    """Loads JSON file associated with dataset.

    :param path: file path
    :return: arrays associated with mappings, features, and labels, respectively
    """
    with open(path, "r") as file:
        data = json.load(file)

    mappings = np.array(data[MAPPING_DATA])
    inputs = np.array(data[INPUT_DATA])
    labels = np.array(data[LABEL_DATA])
    return mappings, inputs, labels


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
    inputs = np.reshape(inputs, newshape=(num_samples, time_steps, num_features))
    inputs = np.expand_dims(inputs, CHANNELS)
    return inputs


def initialize_model(
    input_shape: Sequence[int],
    num_labels: int
) -> tf.keras.Model:
    """Builds and configures model for training.

    :param input_shape: feature data dimensionality
    :param num_labels: number of all possible output categories
    :return: trainable model
    """
    model = build_model(input_shape, num_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"],
                  )
    return model


def build_model(
    input_shape: Sequence[int],
    num_labels: int
) -> tf.keras.Model:
    """Builds a 2D convolutional neural network model composed of three
    convolutional blocks followed by a fully connected block.

    :param input_shape: feature data dimensionality
    :param num_labels: number of all possible output categories
    :return: model with layers added
    """
    model = tf.keras.models.Sequential()

    # Convolutional block 1
    model.add(tf.keras.layers.Input(input_shape))
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convolutional block 2
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Convolutional block 3
    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(2, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())

    # Fully connected block
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_labels, activation="softmax"))
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
    inputs: Sequence[int],
    labels: Sequence[int]
) -> None:
    """Uses trained model to make predictions on test data. Accuracy
    on entire test dataset is displayed and confidence values are
    displayed on a subset of the test dataset.

    :param model: trained model
    :param mappings: mappings data
    :param inputs: test inputs
    :param labels: test labels
    :return: None
    """
    predicted = np.argmax(model.predict(inputs), axis=1)
    correct = sum(predicted == labels)
    accuracy = correct / len(labels)
    print(f"Test set accuracy: {accuracy:.0%}")
    
    for _ in range(10):
        i = random.randrange(len(inputs) - 1)
        prediction = model(np.expand_dims(inputs[i], SAMPLES))
        plt.bar(mappings, tf.nn.softmax(prediction[0]))
        plt.title(f'Predictions for "{mappings[labels[i]]}"')
        plt.xlabel("Labels")
        plt.ylabel("Confidence")
        plt.show()


def main() -> None:
    """Builds, trains, and evaluates a 2D convolutional neural network
    model on a dataset for music genre prediction. Training is done
    incrementally to reduce memory overhead.

    :return: None
    """
    verify_arguments()

    for i, path in enumerate(sys.argv[1:], start=1):
        mappings, inputs, labels = load_data(path)
        inputs = preprocess_data(inputs)
        
        train_inputs, remaining_inputs, \
            train_labels, remaining_labels = train_test_split(inputs,
                                                            labels,
                                                            train_size=0.9
                                                            )
        validate_inputs, test_inputs, \
            validate_labels, test_labels = train_test_split(remaining_inputs,
                                                            remaining_labels,
                                                            test_size=0.5
                                                            )

        if i == 1:
            input_shape = (inputs.shape[TIME], inputs.shape[FEATURES], inputs.shape[CHANNELS])
            model = initialize_model(input_shape, len(mappings))
            model.summary()

        history = model.fit(train_inputs, train_labels, batch_size=32, epochs=35,
                            validation_data=(validate_inputs, validate_labels),
                            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)
                            )
        print(f"Finished training on dataset {i}/{len(sys.argv[1:])}")
    display_training_metrics(history)
    evaluate_model(model, mappings, test_inputs, test_labels)


if __name__ == "__main__":
    main()
