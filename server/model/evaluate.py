"""Evaluates a neural network model on a dataset for music genre prediction."""

from typing import Dict, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

from model.dataset import SAMPLES

np_config.enable_numpy_behavior()


def evaluate_model(
    model: tf.keras.Model,
    inputs: Sequence[int],
    mappings: Sequence[int]
) -> Dict[str, float]:
    """Evaluates model on a dataset.

    :param model: preconfigured model
    :param inputs: array associated with input
    :param mappings: array associated with mappings
    :return: dictionary of genres with each prediction
    """
    input = np.expand_dims(inputs[0], SAMPLES)
    prediction = model(input, training=False)

    prediction_scaled = tf.nn.softmax(prediction[0]).tolist()
    probabilities = create_genres_dictionary(prediction_scaled, mappings)
    return probabilities


def evaluate_model_max(
    model: tf.keras.Model,
    inputs: Sequence[int],
    mappings: Sequence[int]
) -> Dict[str, float]:
    """Evaluates model on a dataset and retrieves the input associated
    with the highest single confidence prediction.

    :param model: preconfigured model
    :param inputs: array associated with input
    :param mappings: array associated with mappings
    :return: dictionary of genres with each prediction
    """
    num_inputs = inputs.shape[SAMPLES]

    confidence_max = 0
    predictions_max = []

    for i in range(num_inputs):
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        prediction_scaled = tf.nn.softmax(prediction[0])
        confidence = np.max(prediction_scaled)

        if confidence > confidence_max:
            confidence_max = confidence
            predictions_max = prediction_scaled.tolist()

    probabilities = create_genres_dictionary(predictions_max, mappings)
    return probabilities


def create_genres_dictionary(
    predictions: Sequence[int],
    mappings: Sequence[int]
) -> Dict[str, float]:
    """Creates a dictionary with probabilities associated with
    each genre.

    :param predictions: list of index-based probabilities
    :param mappings: array associated with mappings
    :return: dictionary of genres with each prediction
    """
    probabilities = {"genres": []}

    for i in range(len(mappings)):
        probabilities["genres"].append({
            "genre": mappings[i],
            "prediction": predictions[i]
        })
    return probabilities
