import evaluate
import numpy as np


def load_accuracy_metric():
    """Loads and returns the accuracy metric."""
    return evaluate.load("accuracy")


def load_f1_metric():
    """Loads and returns the weighted F1 metric."""
    return evaluate.load("f1", average="weighted")


def compute_metrics(eval_pred):
    """Calculates accuracy and F1-score for the given predictions and labels."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy_metric = load_accuracy_metric()
    f1_metric = load_f1_metric()

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_score = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )

    return {
        "accuracy": acc["accuracy"],
        "f1": f1_score["f1"],
    }
