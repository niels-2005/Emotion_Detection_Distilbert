import os

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from model import get_base_model
from tokenizer import load_tokenizer

# Configuration and Constants
BATCH_SIZE = 32
DATASET_CSV_PATH = "./dataset_csvs"
TRAIN_CSV_PATH = os.path.join(DATASET_CSV_PATH, "emotion_train.csv")
VALIDATION_CSV_PATH = os.path.join(DATASET_CSV_PATH, "emotion_validation.csv")
TEST_CSV_PATH = os.path.join(DATASET_CSV_PATH, "emotion_test.csv")


def get_dataset_config():
    return {
        "batch_size": BATCH_SIZE,
        "train_csv_path": TRAIN_CSV_PATH,
        "validation_csv_path": VALIDATION_CSV_PATH,
        "test_csv_path": TEST_CSV_PATH,
    }


def get_data_collator(tokenizer) -> DataCollatorWithPadding:
    """Initializes and returns a data collator with padding for TensorFlow datasets."""
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


def load_csv_as_dataset(csv_path: str) -> Dataset:
    """Loads a CSV file and converts it to a Hugging Face Dataset."""
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)


def get_dataset() -> DatasetDict:
    """Creates a DatasetDict for training, validation, and testing."""
    dataset_config = get_dataset_config()
    return DatasetDict(
        {
            "train": load_csv_as_dataset(dataset_config["train_csv_path"]),
            "validation": load_csv_as_dataset(dataset_config["validation_csv_path"]),
            "test": load_csv_as_dataset(dataset_config["test_csv_path"]),
        }
    )


def preprocess_dataset(samples: dict, tokenizer, truncation: bool) -> dict:
    """Tokenizes datasets and applies truncation."""
    return tokenizer(samples["text"], truncation=truncation)


def get_tokenized_dataset() -> DatasetDict:
    """Loads and tokenizes the dataset, returning a tokenized DatasetDict."""
    dataset = get_dataset()
    tokenizer = load_tokenizer()
    return dataset.map(
        lambda samples: preprocess_dataset(samples, tokenizer, truncation=True),
        batched=True,
    )


def get_tensorflow_datasets(
    return_only_validation: bool = False, return_only_test: bool = False
):
    """Prepares TensorFlow datasets; optionally returns only validation or test set."""
    tokenized_dataset = get_tokenized_dataset()
    base_model = get_base_model()
    tokenizer = load_tokenizer()
    data_collator = get_data_collator(tokenizer)
    dataset_config = get_dataset_config()

    tf_datasets = {
        "train": base_model.prepare_tf_dataset(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=dataset_config["batch_size"],
            collate_fn=data_collator,
        ),
        "validation": base_model.prepare_tf_dataset(
            tokenized_dataset["validation"],
            shuffle=False,
            batch_size=dataset_config["batch_size"],
            collate_fn=data_collator,
        ),
        "test": base_model.prepare_tf_dataset(
            tokenized_dataset["test"],
            shuffle=False,
            batch_size=dataset_config["batch_size"],
            collate_fn=data_collator,
        ),
    }

    if return_only_validation:
        return tf_datasets["validation"]
    elif return_only_test:
        return tf_datasets["test"]
    else:
        return tf_datasets["train"], tf_datasets["validation"], tf_datasets["test"]
