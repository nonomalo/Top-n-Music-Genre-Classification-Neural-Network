"""Builds, trains, and evaluates a 2D convolutional neural network
model on a dataset for music genre prediction.

CL: python3 model.py json-filepath ...
"""

import json
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


def load_data(paths: Sequence[str]) -> Tuple[np.array, np.array, np.array]:
    """Loads JSON files associated with dataset. JSON files must be structured
    identically.

    :param paths: array with file paths
    :return: arrays associated with mappings, features, and labels, respectively
    """
    mappings = []
    inputs = []
    labels = []

    try:
        for i, path in enumerate(paths):
            with open(path, "r") as file:
                data = json.load(file)

            # Identical mappings between files expected
            if i == 0:
                mappings.extend(data["mapping"])
    
            inputs.extend(data["mfcc"])
            labels.extend(data["labels"])
    except FileNotFoundError:
        sys.exit(f"'{path}' is an invalid file path")
    except IndexError:
        sys.exit(f"'{path}' has an incompatible JSON structure")
    
    mappings = np.array(mappings)
    inputs = np.array(inputs)
    labels = np.array(labels)
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
    inputs: Sequence[int],
    labels: Sequence[int]
) -> None:
    """Uses trained model to make predictions on test data.

    :param model: trained model
    :param inputs: test input
    :param labels: test labels
    :return: None
    """
    predicted = np.argmax(model.predict(inputs), axis=1)
    correct = sum(predicted == labels)
    accuracy = correct / len(labels)
    print(f"Test set accuracy: {accuracy:.0%}")


def main() -> None:
    """Builds, trains, and evaluates a 2D convolutional neural network
    model on a dataset for music genre prediction.

    :return: None
    """
    dataset_paths = []

    try:
        for arg in sys.argv[1:]:
            dataset_paths.append(arg)
    except IndexError:
        sys.exit(f"Usage: python3 {sys.argv[0]} json-filepath ...")

    mappings, inputs, labels = load_data(dataset_paths)
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

    input_shape = (inputs.shape[TIME], inputs.shape[FEATURES], inputs.shape[CHANNELS])
    model = build_model(input_shape, len(mappings))

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"],
                  )

    history = model.fit(train_inputs, train_labels, batch_size=32, epochs=35,
                        validation_data=(validate_inputs, validate_labels),
                        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)
                        )
    display_training_metrics(history)
    evaluate_model(model, test_inputs, test_labels)


if __name__ == "__main__":
    main()
