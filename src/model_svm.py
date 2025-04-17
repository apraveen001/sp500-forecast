# src/model_svm.py

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def build_svm(
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale'
) -> SVC:
    """
    Instantiate an SVM classifier with given hyperparameters.

    Args:
        kernel: kernel type ('rbf', 'linear', etc.)
        C: regularization parameter
        gamma: kernel coefficient for 'rbf', 'poly' and 'sigmoid'

    Returns:
        Untrained sklearn.svm.SVC object
    """
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    return clf


def train_svm(
    clf: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> SVC:
    """
    Train the SVM classifier on training data.

    Returns:
        Trained SVC classifier
    """
    clf.fit(X_train, y_train)
    return clf


def evaluate_svm(
    clf: SVC,
    X: np.ndarray,
    y_true: np.ndarray,
    target_names: list = ['Bad', 'Good']
) -> tuple:
    """
    Evaluate classifier performance.

    Returns:
        accuracy: float
        report: str (classification report)
    """
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names)
    return acc, report


if __name__ == '__main__':
    # quick test stub
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=200, n_features=5, random_state=42)
    svm = build_svm()
    svm = train_svm(svm, X_dummy[:160], y_dummy[:160])
    acc, rep = evaluate_svm(svm, X_dummy[160:], y_dummy[160:])
    print(f"Test Accuracy: {acc:.2f}\nReport:\n{rep}")