"""Contains constants associated with the neural network model."""

import tensorflow as tf

SAMPLES = 0
TIME = 1
FEATURES = 2
CHANNELS = 3

MAPPINGS = [
    "International", "Blues", "Jazz", "Classical",
    "Old-Time / Historic", "Country", "Pop", "Rock",
    "Easy Listening", "Soul-RnB", "Electronic",
    "Folk", "Spoken", "Hip-Hop", "Experimental",
    "Instrumental"
]

TRAINING_FRACTION = 0.9
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                     restore_best_weights=True)
]
