import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from get_predictions import get_y_pred
from model_history import load_history

# Configuration and Constants
EVALUATION_FOLDER = "./model_evaluation"
METRICS_AVERAGE = "weighted"
CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TEST_CSV_PATH = "dataset_csvs/emotion_test.csv"


def get_evaluation_config():
    return {
        "evaluation_folder": EVALUATION_FOLDER,
        "metrics_average": METRICS_AVERAGE,
        "classes": CLASSES,
        "test_csv_path": TEST_CSV_PATH,
    }


def load_test_data():
    """Loads test data and separates features and labels."""
    config = get_evaluation_config()
    df_test = pd.read_csv(config["test_csv_path"])
    X_test = df_test["text"].values
    y_true = df_test["label"].values
    return X_test, y_true


def classification_evaluation_pipeline():
    """Runs the complete evaluation pipeline: plots history, classification report,
    confusion matrix, incorrect predictions, and key metrics."""
    X_test, y_true = load_test_data()
    y_pred = get_y_pred()
    model_history = load_history()
    config = get_evaluation_config()

    print("Plotting Model History")
    plot_model_history(history=model_history, save_folder=config["evaluation_folder"])

    print("Printing Classification Report")
    print(
        classification_report(
            y_pred=y_pred, y_true=y_true, target_names=config["classes"]
        )
    )

    report = classification_report(
        y_pred=y_pred, y_true=y_true, output_dict=True, target_names=config["classes"]
    )
    plot_classification_report_with_support(
        report=report, save_folder=config["evaluation_folder"]
    )

    report_df = pd.DataFrame(report)
    report_df.to_csv(f"./model_evaluation/classification_report.csv")

    print("Plotting Confusion Matrix")
    make_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        classes=config["classes"],
        save_folder=EVALUATION_FOLDER,
    )

    print("Analyzing Wrong Predictions")
    df_predictions, wrong_preds = get_wrong_predictions(
        X_test=X_test, y_pred=y_pred, y_true=y_true, classes=config["classes"]
    )
    df_predictions.to_csv(f"./model_evaluation/all_model_predictions.csv", index=False)
    wrong_preds.to_csv(f"./model_evaluation/model_wrong_predictions.csv", index=False)

    print("Calculating Accuracy, F1-Score, Precision, and Recall")
    df_metrics = calculate_metrics(
        y_pred=y_pred, y_true=y_true, average=config["metrics_average"]
    )
    df_metrics.to_csv(f"./model_evaluation/model_metrics_weighted.csv", index=False)


def plot_model_history(history: dict, save_folder: str):
    """Plots training and validation loss, F1 score, accuracy, and learning rate from history."""
    loss, val_loss = history["loss"], history["val_loss"]
    val_accuracy, val_f1 = history["accuracy"], history["f1"]
    lr = history["lr"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    # Loss plot
    plt.subplot(3, 1, 1)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Validation metrics plot
    plt.subplot(3, 1, 2)
    plt.plot(epochs, val_f1, label="Validation F1 - Score")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Validation F1 - Score and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()

    # Learning rate plot
    plt.subplot(3, 1, 3)
    plt.plot(epochs, lr, label="Learning Rate")
    plt.title("Learning Rate over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_folder}/model_history.png")


def plot_classification_report_with_support(report: dict, save_folder: str):
    """Plots classification report with support counts for each class."""
    labels = list(report.keys())[:-3]
    metrics = ["precision", "recall", "f1-score", "support"]
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(data, cmap="coolwarm")
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.title("Classification Report with Support")
    plt.savefig(f"{save_folder}/classification_report.png")


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_folder: str,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
):
    """Creates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if norm else None)

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            ha="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    plt.savefig(f"{save_folder}/confusion_matrix.png")


def get_wrong_predictions(
    X_test: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns DataFrames of all predictions and only incorrect predictions."""
    df_predictions = pd.DataFrame(
        {
            "text": X_test,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_true_classnames": [classes[i] for i in y_true],
            "y_pred_classnames": [classes[i] for i in y_pred],
        }
    )
    df_predictions["pred_correct"] = (
        df_predictions["y_true"] == df_predictions["y_pred"]
    )

    wrong_preds = df_predictions[df_predictions["pred_correct"] == False]
    return df_predictions, wrong_preds


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> pd.DataFrame:
    """Calculates accuracy, F1-score, precision, and recall, and returns them as a DataFrame."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"f1_score_{average}": f1_score(y_true, y_pred, average=average),
        f"precision_{average}": precision_score(y_true, y_pred, average=average),
        f"recall_{average}": recall_score(y_true, y_pred, average=average),
    }
    return pd.DataFrame([metrics])
