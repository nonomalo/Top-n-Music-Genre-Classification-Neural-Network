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
    mappings: Sequence[int],
    display: int,
    labels: Sequence[int] = None
) -> Dict[str, float]:
    """Evaluates model on a dataset and displays predictions for
    a specified number of inputs.

    :param model: preconfigured model
    :param inputs: array associated with input
    :param mappings: array associated with mappings
    :param display: number of predictions to display
    :param labels: array associated with labels, optional
    :return: dictionary of genres with each prediction
    """
    json_dict = {'genres': []}

    for i in range(display):
        input = np.expand_dims(inputs[i], SAMPLES)
        prediction = model(input, training=False)

        # map predictions into a dictionary
        genre_predictions = tf.nn.softmax(prediction[0]).tolist()
        for i in range(len(mappings)):
            json_dict['genres'].append({
                'genre': mappings[i],
                'prediction': genre_predictions[i]
            })

    return json_dict
