import tensorflow as tf

from model import get_trained_model
from tokenizer import load_tokenizer


def predict_class(text: str, model, tokenizer) -> str:
    """Predicts the class label for the given text using the specified model and tokenizer."""
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    predicted_label = model.config.id2label[predicted_class_id]
    return predicted_label


def test_model_with_sample(text: str):
    """Tests the model with the provided text sample."""
    model = get_trained_model()
    tokenizer = load_tokenizer()
    predicted_label = predict_class(text, model, tokenizer)
    print(predicted_label)


if __name__ == "__main__":
    sample_text = "I want to murder my girlfriend"
    test_model_with_sample(text=sample_text)
