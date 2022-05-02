"""Builds, trains, and evaluates a 2D convolutional neural network
model on a dataset for music genre prediction.

CL: python3 model.py <json-filepath>
"""

import json
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

SAMPLES = 0
TIME = 1
MFCCS = 2
CHANNELS = 3


def load_data(path: str) -> Tuple[np.array, np.array, np.array]:
    """Loads JSON file associated with dataset.

    :param path: absolute file path
    :return: arrays associated with mapping, MFCCs, and labels, respectively
    """
    with open(path, 'r') as file:
        data = json.load(file)

    mapping = np.array(data['mapping'])
    inputs = np.array(data['mfcc'])
    labels = np.array(data['labels'])
    return mapping, inputs, labels


def preprocess_data(inputs: np.array) -> np.array:
    """Swaps dimensions representing MFCCs and time while maintaining samples
    dimension and adds an innermost dimension representing channels.

    :param inputs: data of the shape (samples, MFCCs, time)
    :return: data of the shape (samples, time, MFCCs, channels)
    """
    inputs = np.transpose(inputs, (SAMPLES, MFCCS, TIME))
    inputs = np.expand_dims(inputs, CHANNELS)
    return inputs


def build_model(
    input_shape: Tuple[int, int, int],
    num_labels: int
) -> tf.keras.Model:
    """Builds a 2D convolutional neural network model composed of three
    convolutional blocks followed by a fully connected block.

    :param input_shape: feature data dimensionality
    :param num_labels: number of all possible output categories
    :return: model with layers added
    """
    model = models.Sequential()

    # Convolutional block 1
    model.add(layers.Input(input_shape))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPool2D(3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())

    # Convolutional block 2
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPool2D(3, strides=2, padding='same'))
    model.add(layers.BatchNormalization())

    # Convolutional block 3
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPool2D(2, padding='same'))
    model.add(layers.BatchNormalization())

    # Fully connected block
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_labels, activation='softmax'))
    return model


def display_training_metrics(history: tf.keras.callbacks.History) -> None:
    """Displays loss metrics associated with model during training.

    :param history: history object returned by model after training
    :return: None
    """
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.title("Training Loss Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Loss', 'Validation loss'])
    plt.show()


def evaluate_model(
    model: tf.keras.Model,
    inputs: np.array,
    labels: np.array
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
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} <json-filepath>")

    if not os.path.exists(sys.argv[1]):
        sys.exit(f"{sys.argv[1]} is not a valid file path")

    dataset_path = sys.argv[1]
    mapping, inputs, labels = load_data(dataset_path)
    inputs = preprocess_data(inputs)

    # Representative of a single input
    audio_shape = (inputs.shape[TIME],
                   inputs.shape[MFCCS],
                   inputs.shape[CHANNELS]
                   )

    train_inputs, remaining_inputs, \
        train_labels, remaining_labels = train_test_split(inputs,
                                                          labels,
                                                          train_size=0.8
                                                          )
    validate_inputs, test_inputs, \
        validate_labels, test_labels = train_test_split(remaining_inputs,
                                                        remaining_labels,
                                                        test_size=0.5
                                                        )

    model = build_model(audio_shape, len(mapping))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
                  )
    # Training
    history = model.fit(train_inputs, train_labels, batch_size=32, epochs=15,
                        validation_data=(validate_inputs, validate_labels)
                        )
    display_training_metrics(history)
    evaluate_model(model, test_inputs, test_labels)


if __name__ == "__main__":
    main()
