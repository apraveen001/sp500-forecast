# src/evaluate.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from tensorflow.keras.models import load_model

# Assuming these imports are correct relative to your project structure
from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
# Use updated features.py
from features import build_technical_features
# Keep other imports commented out for now
# from model_svm import evaluate_svm
# from strategy import evaluate_strategy, plot_cumulative_returns
# from backtest_strategy import backtest_strategy


def create_windowed_dataset_multivariate(data_array: np.ndarray, window_size: int, target_idx: int):
    """Creates windowed data for LSTM. Target is at target_idx."""
    X, y = [], []
    n_samples = data_array.shape[0]
    if n_samples <= window_size:
        return np.array(X), np.array(y)
    for i in range(n_samples - window_size):
        X.append(data_array[i : i + window_size])
        y.append(data_array[i + window_size, target_idx])
    return np.array(X), np.array(y)


def prepare_test_data_price(window_size: int = 20, ic_window: int = 252, years_of_data: int = 10):
    """
    Loads data, calculates features, filters for last N years,
    EXCLUDES IC features, scales using the specific scaler,
    prepares data for PRICE prediction evaluation.
    Returns X_test, y_test (scaled price), original prices, dates.
    """
    # --- This logic mirrors prepare_data in train.py (10yr, No IC) ---

    # 1. Load & preprocess FULL data first
    print(f"Loading full dataset...")
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df_full = preprocess_prices(df_raw).sort_index()

    # --- Filter for the last N years ---
    end_date = df_full.index.max()
    start_date = end_date - pd.DateOffset(years=years_of_data)
    df = df_full[df_full.index >= start_date].copy()
    print(f"\nFiltering for last {years_of_data} years...")
    print(f"Using data from {df.index.min()} to {df.index.max()}")
    if df.empty:
        raise ValueError(f"No data found for the last {years_of_data} years.")
    # --- End Date Filtering ---

    # 2. ICs - Calculate on filtered data but DO NOT use
    print("Calculating Initial Conditions on filtered data...")
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features - Calculate on filtered data
    print("Calculating Technical Features on filtered data...")
    tech_df = build_technical_features(df)

    # 4. Merge features initially
    df_all_merged = (
        df[['Close', 'Open', 'High', 'Low', 'Volume']]
        .join(tech_df, how='inner')
        .join(ic_df, how='inner')
        .dropna()
    )

    # --- Define columns to KEEP (Exclude ICs) ---
    ic_columns_to_drop = ['phi', 'theta', 'omega', 'alpha', 'beta', 'entropy']
    cols_to_drop_existing = [col for col in ic_columns_to_drop if col in df_all_merged.columns]
    df_selected = df_all_merged.drop(columns=cols_to_drop_existing).copy()
    print(f"Columns selected for scaling in prepare_test_data_price: {list(df_selected.columns)}")

    # --- Identify Target Column ('Close') ---
    try:
        target_col_name = 'Close'
        target_idx = list(df_selected.columns).index(target_col_name)
    except ValueError:
        print(f"ERROR: '{target_col_name}' column not found in df_selected columns!")
        raise

    # 5. Load the FITTED scaler (trained on last N years, selected features)
    scaler_filename = 'models/scaler_10yr.pkl' # Use the correct scaler name
    try:
        scaler = joblib.load(scaler_filename)
        print(f"Loaded scaler object from {scaler_filename}: {scaler}")
        if scaler.n_features_in_ != df_selected.shape[1]:
             raise ValueError(f"Feature mismatch: Data has {df_selected.shape[1]}, Scaler expects {scaler.n_features_in_}")
    except FileNotFoundError:
        print(f"ERROR: {scaler_filename} not found! Run train.py first with 10yr setting.")
        raise
    except Exception as e:
        print(f"Error loading or verifying {scaler_filename}: {e}")
        raise

    # 6. Transform the SELECTED data using the loaded scaler
    data_scaled = scaler.transform(df_selected)

    # --- Store Original Prices for Evaluation ---
    original_prices = df_selected['Close'].values
    original_dates = df_selected.index
    # ---

    # 8. Window the scaled data for PRICE prediction
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 9. Split (80/10/10) - Only need the test portion here
    n = X.shape[0]
    if n == 0:
        raise ValueError("No windowed data created during evaluation preparation.")
    # Test set is the last 10% of the windowed 10-year data
    test_start = int(0.9 * n)

    X_test, y_test = X[test_start:], y[test_start:] # y_test is scaled 'Close' price

    # Get corresponding original prices and dates for the test set
    test_set_start_index_orig = window_size + test_start
    end_index = test_set_start_index_orig + len(y_test)
    if end_index > len(original_prices):
        end_index = len(original_prices)

    y_test_orig_prices = original_prices[test_set_start_index_orig : end_index]
    test_dates = original_dates[test_set_start_index_orig : end_index]

    if len(y_test) != len(y_test_orig_prices):
         min_len_test = min(len(y_test), len(y_test_orig_prices))
         X_test = X_test[:min_len_test]
         y_test = y_test[:min_len_test]
         y_test_orig_prices = y_test_orig_prices[:min_len_test]
         test_dates = test_dates[:min_len_test]

    print(f"Test shapes (10yr): X_test={X_test.shape}, y_test (scaled Close)={y_test.shape}")
    print(f"Test original shapes (10yr): Prices={y_test_orig_prices.shape}, Dates={test_dates.shape}")

    # Return values needed for evaluation
    return (X_test, y_test, # Scaled test data
            y_test_orig_prices, test_dates) # Original test data


def main():
    os.makedirs('plots', exist_ok=True)

    # Prepare test data (using last 10 years, excluding IC features)
    window_size = 20
    ic_window   = 252
    years_data_to_use = 10 # <<<--- Ensure this matches train.py setting
    (X_test, y_test_scaled_price, # LSTM input/output
     y_test_orig_prices, test_dates # Ground truth
     ) = prepare_test_data_price(window_size, ic_window, years_of_data=years_data_to_use)

    # Load the trained LSTM model (trained on 10yr data)
    model_filename = 'models/lstm_price_10yr_model.h5' # Use correct model name
    try:
        lstm = load_model(model_filename, compile=False)
    except Exception as e:
        print(f"Error loading LSTM model {model_filename}: {e}")
        raise

    # Load the trained SVM model (Commented out - likely incompatible)
    # try:
    #     svm = joblib.load('models/svm_model.pkl')
    # except Exception as e:
    #     print(f"Warning: Error loading SVM model models/svm_model.pkl: {e}")
    #     svm = None

    # --- Generate Price Predictions ---
    preds_scaled_flat = lstm.predict(X_test).flatten()

    # --- Inverse Transform Price Predictions ---
    print("\n--- Loading 'Close' Target Params (10yr) & Inverse Transforming ---")
    params_filename = 'models/target_scaler_params_10yr.pkl' # Use correct params name
    try:
        target_params = joblib.load(params_filename)
        target_mean = target_params['mean']
        target_scale = target_params['scale'] # Std Dev of 'Close' price (10yr)
        print(f"Loaded 'Close' Target Mean (10yr): {target_mean:.4f}")
        print(f"Loaded 'Close' Target StdDev (10yr): {target_scale:.4f}")

        preds_orig = (preds_scaled_flat * target_scale) + target_mean
        # y_test_inv = (y_test_scaled_price.flatten() * target_scale) + target_mean # For checking

        print(f"Sample y_test_orig_prices (first 5): {y_test_orig_prices[:5]}")
        print(f"Sample preds_orig (first 5): {preds_orig[:5]}")
        # print(f"Sample y_test inverse transformed (first 5): {y_test_inv[:5]}")

    except FileNotFoundError:
        print(f"ERROR: {params_filename} not found!")
        raise FileNotFoundError("Target 'Close' scaling parameters not found.")
    except Exception as e:
        print(f"An unexpected error occurred during inverse scaling: {e}")
        raise e
    print("--- End Inverse Transform ---\n")


    # --- Evaluate Price Prediction Quality ---
    print("\n--- Evaluating Price Prediction Quality (10yr Data, No IC Features) ---")

    mae = mean_absolute_error(y_test_orig_prices, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig_prices, preds_orig))
    mean_actual = np.mean(y_test_orig_prices)

    if np.isclose(mean_actual, 0): mae_percentage = np.inf
    else: mae_percentage = (mae / mean_actual) * 100

    print(f"Mean Absolute Error (MAE): {mae:.2f} (Price Points)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Price Points)")
    print(f"Mean Actual Price in Test Set: {mean_actual:.2f}")
    print(f"MAE as Percentage of Mean Price: {mae_percentage:.2f}%")

    # Create the plot comparing prices
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_orig_prices, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(test_dates, preds_orig, label='Predicted Prices (LSTM)', color='red', linestyle='--', linewidth=1.5)
    plt.title(f'S&P 500 Test Set ({years_data_to_use}yr Data): Actual vs. Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('S&P 500 Closing Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f'plots/actual_vs_predicted_prices_{years_data_to_use}yr.png' # New filename
    plt.savefig(plot_filename)
    print(f"Saved prediction quality plot to: {plot_filename}")
    plt.show()
    # --- END Price Prediction Evaluation ---


    # --- Optional evaluations (Strategy, SVM) remain commented out ---


if __name__ == '__main__':
    main()
