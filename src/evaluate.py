# src/evaluate.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

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
from model_svm import evaluate_svm
from strategy import evaluate_strategy, plot_cumulative_returns
from backtest_strategy import backtest_strategy


def create_windowed_dataset_multivariate(data_array: np.ndarray, window_size: int, target_idx: int):
    X, y = [], []
    n_samples = data_array.shape[0]
    for i in range(n_samples - window_size):
        X.append(data_array[i : i + window_size])
        y.append(data_array[i + window_size, target_idx])
    return np.array(X), np.array(y)


def build_svm_data(preds: np.ndarray, actuals: np.ndarray):
    pred_ret = (preds[1:] - preds[:-1]) / preds[:-1]
    X_svm = pred_ret.reshape(-1, 1)
    true_ret = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    y_svm = (true_ret > 0).astype(int)
    return X_svm, y_svm


def prepare_test_set(window_size: int = 20, ic_window: int = 252):
    # load and prep
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df = preprocess_prices(df_raw)

    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)
    tech_df = build_technical_features(df)

    df_all = (
        df[['Close']]
        .join(tech_df, how='inner')
        .join(ic_df, how='inner')
        .dropna()
    )

    scaler = joblib.load('models/scaler.pkl')
    data_scaled = scaler.transform(df_all)
    target_idx = list(df_all.columns).index('Close')

    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)
    n = X.shape[0]
    test_start = int(0.9 * n)
    X_test, y_test = X[test_start:], y[test_start:]
    test_dates = df_all.index[window_size + test_start : ]

    return X_test, y_test, target_idx, test_dates


def main():
    os.makedirs('plots', exist_ok=True)

    # prepare test
    window_size = 20
    ic_window   = 252
    X_test, y_test, target_idx, dates = prepare_test_set(window_size, ic_window)

    # load models
    lstm = load_model('models/lstm_model.h5', compile=False)
    svm  = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    scale, min_ = scaler.scale_[target_idx], scaler.min_[target_idx]

    # generate predictions
    preds = lstm.predict(X_test).flatten()
    preds_orig = preds * scale + min_
    y_test_orig = y_test * scale + min_

    # --- Strategy evaluation ---
    strat = evaluate_strategy(preds_orig, y_test_orig)
    print(f"Total strategy return:       {strat['total_return']:.2%}")
    print(f"Average return per trade:    {strat['avg_return_per_trade']:.2%}")
    plot_cumulative_returns(strat['cumulative_returns'], save_path='plots/strategy_cum_returns.png')

    # --- Backtest with filters, costs, holding period ---
    cum_ret = backtest_strategy(
        dates=dates,
        prices=y_test_orig,
        preds=preds_orig,
        threshold=0.002,
        txn_cost=0.0005,
        holding_period=2,
        long_short=True
    )
    
    import pandas as pd

    # ensure cum_ret is a pandas Series of floats
    if not isinstance(cum_ret, pd.Series):
        # try converting list/ndarray
        cum_ret = pd.Series(cum_ret)

    # coerce anything weird to NaN/numeric
    cum_ret = pd.to_numeric(cum_ret, errors='coerce')
    plt.figure()
    cum_ret.plot(title='Backtest Cumulative Returns')
    plt.savefig('plots/backtest_cum_returns.png')
    plt.show()

    # --- SVM evaluation ---
    X_svm, y_svm = build_svm_data(preds_orig, y_test_orig)
    acc, report = evaluate_svm(svm, X_svm, y_svm)
    print(f"SVM Test Accuracy: {acc:.3f}\n")
    print(report)

    # confusion matrix
    cm = confusion_matrix(y_svm, svm.predict(X_svm))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Down','Up'])
    disp.plot(cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()

    # ROC curve
    y_score = svm.decision_function(X_svm)
    fpr, tpr, _ = roc_curve(y_svm, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'--', label='Random')
    plt.title('SVM ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc_curve.png')
    plt.show()


if __name__ == '__main__':
    main()
