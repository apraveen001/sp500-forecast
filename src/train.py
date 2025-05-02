# src/train.py

import os
import joblib
import numpy as np

# Use StandardScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Assuming these imports are correct relative to your project structure
from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
from features import build_technical_features
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
    """Loads data, calculates features, scales, and splits data."""
    # 1. Load & preprocess
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    df = preprocess_prices(df_raw)

    # 2. ICs (ARMA-GARCH + entropy) - KEEPING THESE
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features - KEEPING THESE
    tech_df = build_technical_features(df)

    # 4. Merge, include Close - USING FULL FEATURE SET
    df_all = (
        df[['Close']] # Ensure 'Close' is first if target_idx=0 is assumed
        .join(tech_df, how='inner')
        .join(ic_df, how='inner')
        .dropna()
    )
    print(f"Columns before scaling (Full Features): {list(df_all.columns)}") # Debug print
    print(f"Shape after merging features and dropping NaNs: {df_all.shape}")

    # 5. Scale using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_all)

    # 6. Save the scaler object
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved StandardScaler object to models/scaler.pkl")

    # 7. Save target parameters for inverse transform
    try:
        target_idx = list(df_all.columns).index('Close') # Find index of 'Close'
    except ValueError:
        print("ERROR: 'Close' column not found in df_all columns!")
        raise
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx] # Std Dev for StandardScaler
    joblib.dump({'mean': target_mean, 'scale': target_scale}, 'models/target_scaler_params.pkl')
    print(f"Saved target mean ({target_mean:.2f}) and std dev ({target_scale:.2f}) for target_idx {target_idx}")

    # 8. Window the scaled data
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 9. Split (80/10/10)
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"Data split shapes: Train X={X_train.shape}, Val X={X_val.shape}, Test X={X_test.shape}")

    # Return target_idx as it's needed by evaluate.py, but not directly used in main() here
    return X_train, y_train, X_val, y_val, X_test, y_test, target_idx


def main():
    window_size = 20
    ic_window   = 252

    # Prepare data (scaling happens inside, using full feature set)
    X_train, y_train, X_val, y_val, X_test, y_test, _ = prepare_data(window_size, ic_window)

    # Build the LSTM model (using the updated simpler build_lstm_model function)
    input_shape = (window_size, X_train.shape[2])
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
    print("Saved trained LSTM model to models/lstm_model.h5")

    # Plot training history (optional but recommended)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Training History (Loss - Simpler Model)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/training_history_loss_simpler.png')
        print("Saved training history plot to plots/training_history_loss_simpler.png")
        # plt.show() # Optionally display plot
    except Exception as plot_e:
        print(f"Could not plot training history: {plot_e}")


if __name__ == '__main__':
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    main()
