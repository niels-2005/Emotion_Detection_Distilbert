import os

import tensorflow as tf

from dataset import get_tensorflow_datasets
from model import get_trained_model


def get_test_dataset():
    """Loads and returns the TensorFlow test dataset."""
    return get_tensorflow_datasets(return_only_test=True, return_only_validation=False)


def get_y_pred():
    """Generates and returns predictions from the trained model on the test dataset."""
    test_dataset = get_test_dataset()
    trained_model = get_trained_model()
    y_pred_logits = trained_model.predict(test_dataset).logits
    y_pred = tf.math.argmax(y_pred_logits, axis=-1).numpy()
    return y_pred
