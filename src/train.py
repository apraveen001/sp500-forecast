# src/train.py

import os
import joblib
import numpy as np
import pandas as pd # Import pandas

# Use StandardScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Assuming these imports are correct relative to your project structure
from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
from features import build_technical_features
# Use the simpler model architecture
from model_lstm import build_lstm_model, train_lstm
# Keep SVM imports if you plan to use it later, otherwise remove
# from model_svm import build_svm, train_svm


def create_windowed_dataset_multivariate(data_array: np.ndarray, window_size: int, target_idx: int):
    """Creates windowed data for LSTM."""
    X, y = [], []
    n_samples = data_array.shape[0]
    for i in range(n_samples - window_size):
        X.append(data_array[i : i + window_size])
        y.append(data_array[i + window_size, target_idx])
    return np.array(X), np.array(y)

# Keep build_svm_data if you plan to use SVM later
# def build_svm_data(preds: np.ndarray, actuals: np.ndarray):
#     """Prepares data for SVM based on predicted vs actual returns."""
#     # ... (implementation as before) ...


def prepare_data(window_size: int = 20, ic_window: int = 252):
    """
    Loads data, calculates features, EXCLUDES IC features, scales,
    and splits data.
    """
    # 1. Load & preprocess
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df = preprocess_prices(df_raw) # Should contain OHLCV

    # 2. ICs (ARMA-GARCH + entropy) - Calculate but DO NOT use for now
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features - Calculate these
    tech_df = build_technical_features(df) # Contains MAs, RSI, BBands, LogRet, Vol

    # 4. Merge features initially (including ICs to align dates correctly)
    df_all_merged = (
        df[['Close', 'Open', 'High', 'Low', 'Volume']] # Include OHLCV from base df
        .join(tech_df, how='inner')
        .join(ic_df, how='inner') # Join ICs to align dates/rows
        .dropna() # Drop NaNs resulting from joins and rolling calculations
    )
    print(f"Columns after merging all features: {list(df_all_merged.columns)}")
    print(f"Shape after merging all features and dropping NaNs: {df_all_merged.shape}")

    # --- NEW: Define columns to KEEP (Exclude ICs) ---
    ic_columns_to_drop = ['phi', 'theta', 'omega', 'alpha', 'beta', 'entropy']
    # Create the final DataFrame by dropping the IC columns
    df_selected = df_all_merged.drop(columns=ic_columns_to_drop)

    print(f"\n--- Using Feature Set EXCLUDING ICs ---")
    print(f"Selected columns: {list(df_selected.columns)}")
    print(f"Shape after dropping IC columns: {df_selected.shape}")
    # --- End Feature Selection ---

    # 5. Scale the SELECTED features
    scaler = StandardScaler()
    # Scale only the selected columns
    data_scaled = scaler.fit_transform(df_selected)

    # 6. Save the scaler object (reflects fewer features now)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved StandardScaler object (EXCLUDING IC features) to models/scaler.pkl")

    # 7. Save target parameters for inverse transform
    # IMPORTANT: target_idx is now based on the order in df_selected!
    try:
        # Find index of 'Close' in the df_selected columns
        target_idx = list(df_selected.columns).index('Close')
    except ValueError:
        print("ERROR: 'Close' column not found in df_selected columns!")
        raise
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx] # Std Dev for StandardScaler
    joblib.dump({'mean': target_mean, 'scale': target_scale}, 'models/target_scaler_params.pkl')
    print(f"Saved target mean ({target_mean:.2f}) and std dev ({target_scale:.2f}) for target_idx {target_idx}")

    # 8. Window the scaled data (using data_scaled from df_selected)
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 9. Split (80/10/10)
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"Data split shapes: Train X={X_train.shape}, Val X={X_val.shape}, Test X={X_test.shape}")

    # Return target_idx as it's needed by evaluate.py
    return X_train, y_train, X_val, y_val, X_test, y_test, target_idx


def main():
    window_size = 20
    ic_window   = 252 # Keep this parameter even if features aren't used directly

    # Prepare data (scaling happens inside, EXCLUDING IC features)
    # Note: prepare_data now returns the correct target_idx for the reduced feature set
    X_train, y_train, X_val, y_val, X_test, y_test, target_idx_from_prep = prepare_data(window_size, ic_window)

    # Build the LSTM model (using the simpler build_lstm_model function)
    # Input shape is (window_size, num_features_after_dropping_ICs)
    input_shape = (window_size, X_train.shape[2]) # Get number of features from X_train
    model = build_lstm_model(input_shape=input_shape) # lr defaults to 0.001

    # Train the LSTM model with moderate parameters
    history = train_lstm(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=150,  # Moderate epochs
        batch_size=32,
        early_stopping_patience=15 # Moderate patience
    )

    # Save the trained model
    model.save('models/lstm_model.h5') # Consider using model.save('models/lstm_model.keras')
    print("Saved trained LSTM model (No IC Features) to models/lstm_model.h5")

    # Plot training history (optional but recommended)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Training History (Loss - No IC Features)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/training_history_loss_no_ic.png')
        print("Saved training history plot to plots/training_history_loss_no_ic.png")
        # plt.show() # Optionally display plot
    except Exception as plot_e:
        print(f"Could not plot training history: {plot_e}")


if __name__ == '__main__':
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    main()
