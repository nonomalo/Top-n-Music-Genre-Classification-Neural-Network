"""Evaluates a neural network model on a dataset for music genre prediction.
"""

from typing import Sequence, Dict
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

    # map predictions into a dictionary
    prediction_scaled = tf.nn.softmax(prediction[0]).tolist()
    json_dict = create_genre_dictionary(prediction_scaled, mappings)
    return json_dict


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

    probabilities_max = 0
    predictions_max = []

    for i in range(num_inputs):
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        prediction_scaled = tf.nn.softmax(prediction[0])
        probability = np.max(prediction_scaled)

        if probability > probabilities_max:
            probabilities_max = probability
            predictions_max = prediction_scaled.tolist()

    json_dict = create_genre_dictionary(predictions_max, mappings)
    return json_dict


def create_genre_dictionary(
    predictions: Sequence[int],
    mappings: Sequence[int]
) -> Dict[str, float]:
    """Creates a dictionary with probabilities associated with
    each genre.

    :param inputs: array associated with input
    :param mappings: array associated with mappings
    :return: dictionary of genres with each prediction
    """
    json_dict = {"genres": []}

    for i in range(len(mappings)):
        json_dict["genres"].append({
            "genre": mappings[i],
            "prediction": predictions[i]
        })
    return json_dict
