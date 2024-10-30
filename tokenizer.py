import os

from transformers import AutoTokenizer

CHECKPOINT = "distilbert/distilbert-base-uncased"
TOKENIZER_SAVE_FOLDER = "tokenizer/"


def get_tokenizer_config():
    return {"checkpoint": CHECKPOINT, "tokenizer_save_folder": TOKENIZER_SAVE_FOLDER}


def load_tokenizer():
    """Loads or creates the tokenizer."""
    tokenizer_config = get_tokenizer_config()

    if not os.path.exists(tokenizer_config["tokenizer_save_folder"]):
        os.makedirs(tokenizer_config["tokenizer_save_folder"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["checkpoint"])
        tokenizer.save_pretrained(tokenizer_config["tokenizer_save_folder"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config["tokenizer_save_folder"]
        )
    return tokenizer
