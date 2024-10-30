import os

from tf_keras.optimizers import Adam
from transformers import TFAutoModelForSequenceClassification

MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"
BASE_MODEL_SAVE_FOLDER = "./base_model"
TRAINED_MODEL_SAVE_FOLDER = "./best_model"
NUM_LABELS = 6
ID2LABEL = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
LABEL2ID = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}


def get_model_paths():
    """Returns paths for the base model and trained model."""
    return {
        "checkpoint": MODEL_CHECKPOINT,
        "base_model_save": BASE_MODEL_SAVE_FOLDER,
        "trained_model_save": TRAINED_MODEL_SAVE_FOLDER,
    }


def get_id_label_mappings():
    """Returns mappings for label IDs."""
    return {"num_labels": NUM_LABELS, "id2label": ID2LABEL, "label2id": LABEL2ID}


def get_optimizer():
    """Returns the Adam optimizer with a specified learning rate."""
    return Adam(learning_rate=1e-4)


def get_base_model():
    """Loads or creates the base model."""
    paths = get_model_paths()
    labels = get_id_label_mappings()
    optimizer = get_optimizer()

    if not os.path.exists(paths["base_model_save"]):
        os.makedirs(paths["base_model_save"])
        base_model = TFAutoModelForSequenceClassification.from_pretrained(
            paths["checkpoint"],
            num_labels=labels["num_labels"],
            id2label=labels["id2label"],
            label2id=labels["label2id"],
        )
        base_model.compile(optimizer=optimizer)
        base_model.save_pretrained(paths["base_model_save"])
    else:
        base_model = TFAutoModelForSequenceClassification.from_pretrained(
            paths["base_model_save"],
            num_labels=labels["num_labels"],
            id2label=labels["id2label"],
            label2id=labels["label2id"],
        )
        base_model.compile(optimizer=optimizer)

    return base_model


def save_trained_model(model):
    """Saves the trained model to the specified path."""
    paths = get_model_paths()

    if not os.path.exists(paths["trained_model_save"]):
        os.makedirs(paths["trained_model_save"])

    model.save_pretrained(paths["trained_model_save"])


def get_trained_model():
    """Loads the trained model from the specified path."""
    paths = get_model_paths()
    optimizer = get_optimizer()

    loaded_model = TFAutoModelForSequenceClassification.from_pretrained(
        paths["trained_model_save"]
    )
    loaded_model.compile(optimizer=optimizer)
    return loaded_model
