import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def print_report(y_true, y_pred):
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
