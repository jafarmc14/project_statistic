import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def subjectwise_split(X, y, groups, test_size=0.2, random_state=42):
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], groups[train_idx], groups[test_idx]

def print_report(y_true, y_pred):
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))