from transformers import TFAutoModelForSequenceClassification
import os 


CHECKPOINT = "distilbert/distilbert-base-uncased"
SAVE_PATH = "base_model/"
NUM_LABELS = 6
ID2LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
LABEL2ID = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}


def load_base_model():
    if not os.path.exists(SAVE_PATH): 
        os.makedirs(SAVE_PATH)
        base_model = TFAutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
        base_model.save_pretrained(SAVE_PATH)
    else:
        base_model = TFAutoModelForSequenceClassification.from_pretrained(SAVE_PATH, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
    return base_model