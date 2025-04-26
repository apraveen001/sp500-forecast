# src/train.py

import os
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
from features import build_technical_features
from model_lstm import build_lstm_model, train_lstm
from model_svm import build_svm, train_svm


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
    y_svm = (true_ret > 0).astype(int)
    return X_svm, y_svm


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
    data_scaled = scaler.fit_transform(df_all)

    # persist scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 6. Window
    target_idx = list(df_all.columns).index('Close')
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 7. Split (80/10/10)
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test, target_idx


def main():
    window_size = 20
    ic_window   = 252

    X_train, y_train, X_val, y_val, X_test, y_test, target_idx = prepare_data(window_size, ic_window)

    # 8. Build & train LSTM
    model = build_lstm_model((window_size, X_train.shape[2]))
    train_lstm(model, X_train, y_train, X_val, y_val)
    model.save('models/lstm_model.h5')

    # 9. Invert-scaling helpers
    scaler = joblib.load('models/scaler.pkl')
    scale, min_ = scaler.scale_[target_idx], scaler.min_[target_idx]

    # 10. Generate test predictions
    preds = model.predict(X_test).flatten()
    preds_orig = preds * scale + min_
    y_test_orig = y_test * scale + min_

    # 11. Train & save SVM
    X_svm, y_svm = build_svm_data(preds_orig, y_test_orig)
    svm = build_svm()
    train_svm(svm, X_svm, y_svm)
    joblib.dump(svm, 'models/svm_model.pkl')


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    main()
