# src/train.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from tensorflow.keras.models import load_model

from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
from features import build_technical_features
from model_lstm import build_lstm_model, train_lstm
from model_svm import build_svm, train_svm, evaluate_svm


def create_windowed_dataset_multivariate(data_array: np.ndarray, window_size: int, target_idx: int):
    X, y = [], []
    n_samples = data_array.shape[0]
    for i in range(n_samples - window_size):
        X.append(data_array[i : i + window_size])
        y.append(data_array[i + window_size, target_idx])
    return np.array(X), np.array(y)


def build_svm_data(preds: np.ndarray, actuals: np.ndarray):
    """
    Features: predicted returns
    Labels: actual up/down based on real returns
    """
    pred_ret = (preds[1:] - preds[:-1]) / preds[:-1]
    X_svm = pred_ret.reshape(-1, 1)

    true_ret = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    labels = (true_ret > 0).astype(int)

    return X_svm, labels


def prepare_data(window_size: int = 20, ic_window: int = 252):
    # 1. Load & preprocess
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df = preprocess_prices(df_raw)

    # 2. ICs (ARMA-GARCH + entropy)
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features
    tech_df = build_technical_features(df)

    # 4. Merge, include Close
    df_all = (
        df[['Close']]
        .join(tech_df, how='inner')
        .join(ic_df, how='inner')
        .dropna()
    )

    # 5. Scale
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(df_all),
        index=df_all.index,
        columns=df_all.columns
    )

    # Save scaler for evaluation
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 6. Window
    feat_array = data_scaled.values
    target_idx = df_all.columns.get_loc('Close')
    X, y = create_windowed_dataset_multivariate(feat_array, window_size, target_idx)

    # 7. Split (80/10/10)
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test, target_idx, scaler


def main():
    os.makedirs('plots', exist_ok=True)

    # Prepare
    window_size = 20
    ic_window = 252
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test,
     target_idx, scaler) = prepare_data(window_size, ic_window)

    # 8. LSTM
    model = build_lstm_model((window_size, X_train.shape[2]))
    train_lstm(model, X_train, y_train, X_val, y_val)

    # save LSTM
    model.save('models/lstm_model.h5')

    # 9. LSTM → test preds
    test_preds = model.predict(X_test).flatten()
    scale, min_ = scaler.scale_[target_idx], scaler.min_[target_idx]
    test_preds_orig = test_preds * scale + min_
    y_test_orig     = y_test   * scale + min_

    # 10. SVM
    svm = build_svm()
    X_svm, y_svm = build_svm_data(test_preds_orig, y_test_orig)
    svm = train_svm(svm, X_svm, y_svm)

    # save SVM
    joblib.dump(svm, 'models/svm_model.pkl')

    # metrics
    acc, report = evaluate_svm(svm, X_svm, y_svm)
    print(f"SVM Test Accuracy: {acc:.3f}\n")
    print(report)

    # 11. Confusion matrix
    cm = confusion_matrix(y_svm, svm.predict(X_svm))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Bad','Good'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()

    # 12. ROC curve
    y_score = svm.decision_function(X_svm)
    fpr, tpr, _ = roc_curve(y_svm, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'--', label='Random')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc_curve.png')
    plt.show()


if __name__ == '__main__':
    main()
