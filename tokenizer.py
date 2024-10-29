from transformers import AutoTokenizer
import os

CHECKPOINT = "distilbert/distilbert-base-uncased"
SAVE_PATH = "tokenizer/"


def load_tokenizer():
    if not os.path.exists(SAVE_PATH): 
        os.makedirs(SAVE_PATH)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        tokenizer.save_pretrained(SAVE_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    return tokenizer