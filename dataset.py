from datasets import load_dataset
from tokenizer import load_tokenizer

TOKENIZER = load_tokenizer()
TRUNCATION = True 

def preprocess_dataset(samples):
    return TOKENIZER(samples["text"], truncation=TRUNCATION)


def get_preprocessed_dataset():
    dataset = load_dataset("dair-ai/emotion", "split").shuffle()
    tokenized_dataset = dataset.map(preprocess_dataset, batched=True)
    return tokenized_dataset








