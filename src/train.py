# src/train.py

import os
import joblib
import numpy as np
import pandas as pd

# Use StandardScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Assuming these imports are correct relative to your project structure
from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
# Use the updated features.py
from features import build_technical_features
# Use the simpler model architecture
from model_lstm import build_lstm_model, train_lstm
# Keep SVM imports if you plan to use it later, otherwise remove
# from model_svm import build_svm, train_svm


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


def prepare_data(window_size: int = 20, ic_window: int = 252, years_of_data: int = 10):
    """
    Loads data, calculates features, filters for the last N years,
    EXCLUDES IC features, scales, prepares data for PRICE prediction,
    and splits data.
    """
    # 1. Load & preprocess FULL data first
    print(f"Loading full dataset...")
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df_full = preprocess_prices(df_raw).sort_index() # Ensure sorted
    print(f"Full dataset loaded. Range: {df_full.index.min()} to {df_full.index.max()}")

    # --- NEW: Filter for the last N years ---
    end_date = df_full.index.max()
    start_date = end_date - pd.DateOffset(years=years_of_data)
    df = df_full[df_full.index >= start_date].copy() # Use .copy()
    print(f"\nFiltering for last {years_of_data} years...")
    print(f"Using data from {df.index.min()} to {df.index.max()}")
    if df.empty:
        raise ValueError(f"No data found for the last {years_of_data} years.")
    # --- End Date Filtering ---

    # 2. ICs - Calculate on the filtered data
    print("Calculating Initial Conditions on filtered data...")
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features - Calculate on the filtered data
    print("Calculating Technical Features on filtered data...")
    tech_df = build_technical_features(df) # Assumes features.py uses df passed

    # 4. Merge features initially (use the filtered df)
    df_all_merged = (
        df[['Close', 'Open', 'High', 'Low', 'Volume']] # Base columns from filtered df
        .join(tech_df, how='inner')
        .join(ic_df, how='inner')
        # Drop NaNs AFTER joining all potential features
        .dropna()
    )
    print(f"Columns after merging all features: {list(df_all_merged.columns)}")
    print(f"Shape after merging features and dropping NaNs: {df_all_merged.shape}")
    if df_all_merged.empty:
        raise ValueError("DataFrame is empty after merging features and dropping NaNs. Check window sizes vs data length.")

    # --- Define columns to KEEP (Exclude ICs) ---
    ic_columns_to_drop = ['phi', 'theta', 'omega', 'alpha', 'beta', 'entropy']
    cols_to_drop_existing = [col for col in ic_columns_to_drop if col in df_all_merged.columns]
    df_selected = df_all_merged.drop(columns=cols_to_drop_existing).copy() # Use copy()

    print(f"\n--- Using Feature Set EXCLUDING ICs ---")
    print(f"Selected columns: {list(df_selected.columns)}")
    print(f"Shape after dropping IC columns: {df_selected.shape}")
    # --- End Feature Selection ---

    # --- Identify Target Column ('Close') ---
    try:
        target_col_name = 'Close'
        target_idx = list(df_selected.columns).index(target_col_name)
        print(f"Target column for prediction is '{target_col_name}' at index {target_idx}")
    except ValueError:
        print(f"ERROR: '{target_col_name}' column not found in df_selected columns!")
        raise
    # ---

    # 5. Scale the SELECTED features (from the last N years)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_selected)

    # 6. Save the scaler object (trained on last N years)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler_10yr.pkl') # New name
    print("Saved StandardScaler object (10yr data, No IC) to models/scaler_10yr.pkl")

    # 7. Save target parameters (MEAN and STD DEV of 'Close' from last N years)
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx] # Std Dev for StandardScaler
    joblib.dump({'mean': target_mean, 'scale': target_scale}, 'models/target_scaler_params_10yr.pkl') # New name
    print(f"Saved target 'Close' mean ({target_mean:.2f}) and std dev ({target_scale:.2f}) for target_idx {target_idx} (10yr data)")

    # --- Store Original Prices for Evaluation ---
    original_prices = df_selected['Close'].values
    original_dates = df_selected.index
    # ---

    # 8. Window the scaled data for PRICE prediction
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 9. Split (80/10/10) - Applied to the last N years data
    n = X.shape[0]
    if n == 0:
        raise ValueError("No windowed data created. Check input data and window size.")
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:] # y_test is scaled 'Close' price

    # Split original prices/dates - align with y_test
    test_set_start_index_orig = window_size + val_end
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

    print(f"Data split shapes (10yr): Train X={X_train.shape}, Val X={X_val.shape}, Test X={X_test.shape}")
    print(f"Test shapes (10yr): y_test (scaled Close)={y_test.shape}, y_test_orig_prices={y_test_orig_prices.shape}, test_dates={test_dates.shape}")

    # Return values needed for training and evaluation
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            y_test_orig_prices, test_dates) # Removed log returns


def main():
    window_size = 20
    ic_window   = 252
    years_data_to_use = 10 # <<<--- Set number of years here

    # Prepare data (scaling happens inside, using last N years, excluding IC features)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     _, _) = prepare_data(window_size, ic_window, years_of_data=years_data_to_use)

    # Build the LSTM model (using the simpler architecture)
    input_shape = (window_size, X_train.shape[2])
    # Use the lower learning rate
    model = build_lstm_model(input_shape=input_shape, lr=0.0005)

    # Train the LSTM model to predict SCALED 'Close' PRICE
    history = train_lstm(
        model,
        X_train, y_train, # y_train is scaled 'Close'
        X_val, y_val,     # y_val is scaled 'Close'
        epochs=150,
        batch_size=32,
        early_stopping_patience=15
    )

    # Save the trained model
    model.save('models/lstm_price_10yr_model.h5') # New model name
    print(f"Saved trained LSTM model ({years_data_to_use}yr data, No IC) to models/lstm_price_10yr_model.h5")

    # Plot training history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Model Training History (Loss - {years_data_to_use}yr Data, No IC)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/training_history_loss_{years_data_to_use}yr.png') # New plot name
        print(f"Saved training history plot to plots/training_history_loss_{years_data_to_use}yr.png")
        # plt.show()
    except Exception as plot_e:
        print(f"Could not plot training history: {plot_e}")


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    main()
