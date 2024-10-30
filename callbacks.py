from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from transformers.keras_callbacks import KerasMetricCallback

from dataset import get_tensorflow_datasets
from evaluate_during_training import compute_metrics

VERBOSE = 1
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 10
MONITOR_METRIC = "val_loss"
RESTORE_BEST_WEIGHTS = True


def get_callbacks_config():
    """
    Returns configuration settings for training callbacks.
    """
    return {
        "verbose": VERBOSE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "monitor_metric": MONITOR_METRIC,
        "restore_best_weights": True,
    }


def get_validation_dataset():
    """
    Return the Validation Dataset for Evaluation during Training
    """
    return get_tensorflow_datasets(return_only_validation=True, return_only_test=False)


def get_callbacks() -> list:
    """
    Creates and returns a list of training callbacks.
    """
    callbacks_config = get_callbacks_config()

    validation_dataset = get_validation_dataset()

    early_stopping = EarlyStopping(
        patience=callbacks_config["early_stopping_patience"],
        monitor=callbacks_config["monitor_metric"],
        verbose=callbacks_config["verbose"],
        restore_best_weights=callbacks_config["restore_best_weights"],
    )
    reduce_lr_on_plateau = ReduceLROnPlateau(
        patience=callbacks_config["reduce_lr_patience"],
        monitor=callbacks_config["monitor_metric"],
        verbose=callbacks_config["verbose"],
    )
    eval_callback = KerasMetricCallback(
        metric_fn=compute_metrics, eval_dataset=validation_dataset
    )

    return [early_stopping, reduce_lr_on_plateau, eval_callback]
