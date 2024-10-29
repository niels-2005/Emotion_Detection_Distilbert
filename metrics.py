import evaluate 
import numpy as np


F1SCORE = evaluate.load("f1")


def compute_metrics(eval_pred):
    preds, labels = eval_pred 
    preds = np.argmax(preds, axis=1)
    return F1SCORE.compute(predictions=preds, references=labels)
