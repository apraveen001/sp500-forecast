# src/train.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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


def main():
    # 1. Load & preprocess data
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df = preprocess_prices(df_raw)

    # 2. Initial conditions (ARMA-GARCH + entropy)
    ic_df = rolling_initial_conditions(df['Close'], window=252)

    # 3. Technical features
    tech_df = build_technical_features(df)

    # 4. Merge and clean
    df_all = tech_df.join(ic_df, how='inner').dropna()

    # 5. Scale features
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(df_all),
        index=df_all.index,
        columns=df_all.columns,
    )

    # 6. Create windowed datasets
    window_size = 20
    feat_array = data_scaled.values
    target_idx = df_all.columns.get_loc('Close')
    X, y = create_windowed_dataset_multivariate(feat_array, window_size, target_idx)

    # 7. Train/val/test split
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    # 8. Build & train LSTM
    model = build_lstm_model((window_size, X.shape[2]))
    train_lstm(model, X_train, y_train, X_val, y_val)

    # 9. Predict on val/test
    val_preds = model.predict(X_val).flatten()
    test_preds = model.predict(X_test).flatten()

    # 10. Invert scaling for Close
    col_idx = target_idx
    scale = scaler.scale_[col_idx]
    min_  = scaler.min_[col_idx]

    y_val_orig   = (y_val - min_) / scale
    val_preds_orig  = (val_preds - min_) / scale
    y_test_orig  = (y_test - min_) / scale
    test_preds_orig = (test_preds - min_) / scale

    # 11. Prepare SVM data (directional returns)
    def build_svm_data(preds, actuals):
        ret = (preds[1:] - preds[:-1]) / preds[:-1]
        labels = (ret > 0).astype(int)
        # Single-feature (predicted return)
        X_svm = ret.reshape(-1, 1)
        return X_svm, labels

    X_val_svm, y_val_svm     = build_svm_data(val_preds_orig, y_val_orig)
    X_test_svm, y_test_svm   = build_svm_data(test_preds_orig, y_test_orig)

    # 12. Train & evaluate SVM
    svm = build_svm()
    svm = train_svm(svm, X_val_svm, y_val_svm)
    acc, report = evaluate_svm(svm, X_test_svm, y_test_svm)

    print(f"SVM Test Accuracy: {acc:.3f}\n")
    print(report)


if __name__ == '__main__':
    main()