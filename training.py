import os

from callbacks import get_callbacks
from dataset import get_tensorflow_datasets
from evaluate_after_training import classification_evaluation_pipeline
from model import get_base_model, save_trained_model
from model_history import save_history

EPOCHS = 100
EVALUATION_FOLDER = "./model_evaluation"


def get_training_config():
    return {"epochs": EPOCHS, "evaluation_folder": EVALUATION_FOLDER}


def train_model():
    base_model = get_base_model()
    train_dataset, validation_dataset, test_dataset = get_tensorflow_datasets()
    callbacks = get_callbacks()
    training_config = get_training_config()

    history = base_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=training_config["epochs"],
        callbacks=callbacks,
    )

    if not os.path.exists(training_config["evaluation_folder"]):
        os.makedirs(training_config["evaluation_folder"])

    save_history(history=history)

    save_trained_model(model=base_model)

    classification_evaluation_pipeline()


if __name__ == "__main__":
    train_model()
