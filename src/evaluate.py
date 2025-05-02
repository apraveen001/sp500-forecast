# src/evaluate.py

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error
)
from tensorflow.keras.models import load_model

# Assuming these imports are correct relative to your project structure
from data_utils import load_sp500_csv, preprocess_prices
from ic_estimator import rolling_initial_conditions
from features import build_technical_features
from model_svm import evaluate_svm # Keep for potential future use
from strategy import evaluate_strategy, plot_cumulative_returns # Keep for potential future use
from backtest_strategy import backtest_strategy # Keep for potential future use


def create_windowed_dataset_multivariate(data_array: np.ndarray, window_size: int, target_idx: int):
    """Creates windowed data for LSTM."""
    X, y = [], []
    n_samples = data_array.shape[0]
    for i in range(n_samples - window_size):
        X.append(data_array[i : i + window_size])
        y.append(data_array[i + window_size, target_idx])
    return np.array(X), np.array(y)


def build_svm_data(preds: np.ndarray, actuals: np.ndarray):
    """Prepares data for SVM based on predicted vs actual returns."""
    # Calculate predicted returns (handle potential division by zero)
    preds_shifted = preds[:-1]
    valid_preds_idx = preds_shifted != 0
    pred_ret = np.full_like(preds_shifted, np.nan) # Initialize with NaN
    pred_ret[valid_preds_idx] = (preds[1:][valid_preds_idx] - preds_shifted[valid_preds_idx]) / preds_shifted[valid_preds_idx]

    # Calculate true returns (handle potential division by zero)
    actuals_shifted = actuals[:-1]
    valid_actuals_idx = actuals_shifted != 0
    true_ret = np.full_like(actuals_shifted, np.nan) # Initialize with NaN
    true_ret[valid_actuals_idx] = (actuals[1:][valid_actuals_idx] - actuals_shifted[valid_actuals_idx]) / actuals_shifted[valid_actuals_idx]

    # Align and remove NaNs resulting from calculation or original data
    min_len = min(len(pred_ret), len(true_ret))
    pred_ret = pred_ret[:min_len]
    true_ret = true_ret[:min_len]

    valid_idx = ~np.isnan(pred_ret) & ~np.isnan(true_ret)
    X_svm = pred_ret[valid_idx].reshape(-1, 1)
    y_svm = (true_ret[valid_idx] > 0).astype(int)

    return X_svm, y_svm


def prepare_test_set(window_size: int = 20, ic_window: int = 252):
    """
    Loads data, calculates features, EXCLUDES IC features, scales,
    and prepares test split. MIRRORS THE LOGIC IN train.py for feature selection.
    """
    # 1. Load & preprocess
    df_raw = load_sp500_csv('data/sp500_20_years.csv')
    # Ensure preprocess_prices keeps needed columns (OHLCV)
    df = preprocess_prices(df_raw)

    # 2. ICs (ARMA-GARCH + entropy) - Calculate but will be dropped later
    ic_df = rolling_initial_conditions(df['Close'], window=ic_window)

    # 3. Technical features - Calculate these
    tech_df = build_technical_features(df) # Contains MAs, RSI, BBands, LogRet, Vol

    # 4. Merge features initially (including ICs to align dates correctly)
    df_all_merged = (
        # Make sure base df has OHLCV if needed by tech_df or final selection
        df[['Close', 'Open', 'High', 'Low', 'Volume']]
        .join(tech_df, how='inner')
        .join(ic_df, how='inner') # Join ICs to align dates/rows
        .dropna() # Drop NaNs resulting from joins and rolling calculations
    )
    print(f"Columns after merging all features in prepare_test_set: {list(df_all_merged.columns)}")

    # --- NEW: Define columns to KEEP (Exclude ICs) ---
    # --- This MUST match the columns used for fitting the scaler in train.py ---
    ic_columns_to_drop = ['phi', 'theta', 'omega', 'alpha', 'beta', 'entropy']
    # Create the final DataFrame by dropping the IC columns
    # Use .copy() to avoid SettingWithCopyWarning if modifying df_selected later
    df_selected = df_all_merged.drop(columns=ic_columns_to_drop).copy()

    print(f"Columns selected for scaling in prepare_test_set: {list(df_selected.columns)}")
    print(f"Shape after dropping IC columns: {df_selected.shape}")
    # --- End Feature Selection ---


    # 5. Load the FITTED scaler (expects features from df_selected)
    try:
        scaler = joblib.load('models/scaler.pkl')
        print(f"Loaded scaler object: {scaler}")
        # Verify scaler expects the same number of features as df_selected
        if scaler.n_features_in_ != df_selected.shape[1]:
             print(f"ERROR: Number of features in data ({df_selected.shape[1]}) does not match scaler's expected features ({scaler.n_features_in_})!")
             raise ValueError("Feature mismatch between evaluation data and loaded scaler.")
    except FileNotFoundError:
        print("ERROR: models/scaler.pkl not found! Run train.py first.")
        raise
    except Exception as e:
        print(f"Error loading scaler.pkl: {e}")
        raise

    # 6. Transform the SELECTED data using the loaded scaler
    data_scaled = scaler.transform(df_selected)
    print(f"Data scaled shape: {data_scaled.shape}")

    # 7. Find target index ('Close') based on df_selected columns
    try:
        # Find index of 'Close' in the columns that were actually scaled
        target_idx = list(df_selected.columns).index('Close')
        print(f"Using target_idx: {target_idx} (based on selected columns)")
    except ValueError:
        print("ERROR: 'Close' column not found in df_selected columns!")
        raise

    # 8. Create windowed data from the scaled, selected features
    X, y = create_windowed_dataset_multivariate(data_scaled, window_size, target_idx)

    # 9. Split into test set
    n = X.shape[0]
    test_start = int(0.9 * n) # Use last 10% for testing
    X_test, y_test = X[test_start:], y[test_start:]

    # 10. Get corresponding dates for the test set
    # Align dates based on the final length of y_test, using the index from df_selected
    test_dates = df_selected.index[-len(y_test):]

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, test_dates length: {len(test_dates)}")

    # Return the correct target_idx based on the selected features
    return X_test, y_test, target_idx, test_dates


# --- Main function remains largely the same ---
# --- It will now receive the correct X_test, y_test, target_idx ---
# --- based on the reduced feature set ---
def main():
    os.makedirs('plots', exist_ok=True)

    # Prepare test data (now excludes IC features before scaling)
    window_size = 20
    ic_window   = 252 # ic_window still needed for calculation, even if dropped later
    # target_idx returned here is now correct for the data used
    X_test, y_test, target_idx, dates = prepare_test_set(window_size, ic_window)

    # Load the trained LSTM model
    try:
        # This model was trained on data corresponding to the reduced features
        lstm = load_model('models/lstm_model.h5', compile=False)
    except Exception as e:
        print(f"Error loading LSTM model models/lstm_model.h5: {e}")
        raise

    # Load the trained SVM model (keep for potential future use)
    # Note: This SVM was likely trained on data derived from previous models/features
    # Its performance might be poor until retrained on data derived from this LSTM's predictions
    try:
        svm = joblib.load('models/svm_model.pkl')
    except Exception as e:
        print(f"Warning: Error loading SVM model models/svm_model.pkl: {e}")
        svm = None # Set to None if loading fails

    # --- Generate predictions from LSTM ---
    # LSTM input X_test now has the correct number of features (14)
    preds_scaled_flat = lstm.predict(X_test).flatten()
    y_test_scaled_flat = y_test.flatten() # Use the prepared y_test target

    # --- Correct Inverse Transform using StandardScaler params ---
    print("\n--- Loading Target Params & Inverse Transforming (StandardScaler) ---")
    try:
        target_params = joblib.load('models/target_scaler_params.pkl')
        target_mean = target_params['mean']
        target_scale = target_params['scale'] # This is Std Dev
        print(f"Loaded Target Mean: {target_mean:.4f}")
        print(f"Loaded Target StdDev: {target_scale:.4f}")
        # The target_idx used here should match the one saved with these params

        # Manual Inverse Transform for StandardScaler: value = (scaled_value * std_dev) + mean
        preds_orig = (preds_scaled_flat * target_scale) + target_mean
        y_test_orig = (y_test_scaled_flat * target_scale) + target_mean

        # Check if results look like prices
        print(f"Sample y_test_orig (first 5): {y_test_orig[:5]}")
        print(f"Sample preds_orig (first 5): {preds_orig[:5]}")

    except FileNotFoundError:
        print("ERROR: models/target_scaler_params.pkl not found! Make sure train.py was run successfully after changes.")
        raise FileNotFoundError("Target scaling parameters not found.")
    except Exception as e:
        print(f"An unexpected error occurred during inverse scaling: {e}")
        raise e

    print("--- End Inverse Transform ---\n")
    # --- End replacement ---

    # --- Evaluate Raw Prediction Quality ---
    print("\n--- Evaluating Raw LSTM Prediction Quality ---")

    # Calculate error metrics
    mae = mean_absolute_error(y_test_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mean_actual = np.mean(y_test_orig)

    # Avoid division by zero if mean_actual is zero or close to it
    if np.isclose(mean_actual, 0):
        mae_percentage = np.inf # Or handle as appropriate
    else:
        mae_percentage = (mae / mean_actual) * 100 # MAE as % of average price

    print(f"Mean Absolute Error (MAE): {mae:.2f} (Price Points)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Price Points)")
    print(f"Mean Actual Price in Test Set: {mean_actual:.2f}")
    print(f"MAE as Percentage of Mean Price: {mae_percentage:.2f}%")

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_test_orig, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(dates, preds_orig, label='Predicted Prices (LSTM)', color='red', linestyle='--', linewidth=1.5)
    plt.title('S&P 500 Test Set: Actual vs. Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('S&P 500 Closing Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    # Save the plot
    plot_filename = 'plots/actual_vs_predicted_prices_no_ic.png' # Changed filename
    plt.savefig(plot_filename)
    print(f"Saved prediction quality plot to: {plot_filename}")
    plt.show() # Display the plot as well
    # --- END Raw Prediction Evaluation ---


    # --- Optional: Strategy Evaluation (Commented out for now) ---
    # print("\n--- Evaluating Strategy Performance ---")
    # try:
    #     # Using the simple strategy (prediction momentum)
    #     strat = evaluate_strategy(preds_orig, y_test_orig) # Ensure this function handles potential NaNs/zeros
    #     print(f"Simple Strategy Total Return:       {strat['total_return']:.2%}")
    #     print(f"Simple Strategy Avg Return/Trade:   {strat['avg_return_per_trade']:.4%}") # More precision might be needed
    #     plot_cumulative_returns(strat['cumulative_returns'], save_path='plots/strategy_cum_returns.png')
    # except Exception as e:
    #     print(f"Error during simple strategy evaluation: {e}")

    # try:
    #     # Using the backtest strategy (more complex)
    #     cum_ret = backtest_strategy(
    #         dates=dates,
    #         prices=y_test_orig,
    #         preds=preds_orig,
    #         threshold=0.0,      # Keep threshold low initially
    #         txn_cost=0.0005,    # Example transaction cost
    #         holding_period=1,   # Example holding period
    #         long_short=False    # Example: Long only
    #     )

    #     if cum_ret is not None and not cum_ret.empty:
    #         print(f"\nBacktest Strategy Final Return: {cum_ret.iloc[-1]:.2%}")
    #         print(f"Backtest generated {len(cum_ret)} data points.")
    #         print(f"First few backtest returns:\n{cum_ret.head()}")
    #         print(f"Last few backtest returns:\n{cum_ret.tail()}")

    #         # Plotting
    #         plt.figure(figsize=(14, 7))
    #         cum_ret.plot(title='Backtest Cumulative Returns')
    #         plt.xlabel('Entry Date')
    #         plt.ylabel('Cumulative Return')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.savefig('plots/backtest_cum_returns.png')
    #         plt.show()
    #     else:
    #         print("Backtest result is empty or None.")

    # except Exception as e:
    #     print(f"Error during backtest strategy evaluation: {e}")
    # --- End Optional Strategy Evaluation ---


    # --- Optional: SVM Evaluation (Commented out for now) ---
    # print("\n--- Evaluating SVM Performance ---")
    # if svm is not None:
    #     try:
    #         X_svm, y_svm = build_svm_data(preds_orig, y_test_orig)
    #         if len(X_svm) > 0:
    #             acc, report = evaluate_svm(svm, X_svm, y_svm)
    #             print(f"SVM Test Accuracy: {acc:.3f}\n")
    #             print(report)

    #             # Confusion matrix
    #             cm = confusion_matrix(y_svm, svm.predict(X_svm))
    #             disp = ConfusionMatrixDisplay(cm, display_labels=['Down','Up'])
    #             disp.plot(cmap='Blues')
    #             plt.title('SVM Confusion Matrix')
    #             plt.savefig('plots/confusion_matrix.png')
    #             plt.show()

    #             # ROC curve
    #             try:
    #                 y_score = svm.decision_function(X_svm)
    #                 fpr, tpr, _ = roc_curve(y_svm, y_score)
    #                 roc_auc = auc(fpr, tpr)
    #                 plt.figure(figsize=(7, 7))
    #                 plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    #                 plt.plot([0,1],[0,1],'--', label='Random')
    #                 plt.title('SVM ROC Curve')
    #                 plt.xlabel('False Positive Rate')
    #                 plt.ylabel('True Positive Rate')
    #                 plt.legend(loc='lower right')
    #                 plt.grid(True)
    #                 plt.savefig('plots/roc_curve.png')
    #                 plt.show()
    #             except Exception as roc_e:
    #                 print(f"Could not generate ROC curve: {roc_e}")
    #         else:
    #             print("Not enough data to build SVM input after processing returns.")

    #     except Exception as e:
    #         print(f"Error during SVM evaluation: {e}")
    # else:
    #     print("SVM model not loaded, skipping SVM evaluation.")
    # --- End Optional SVM Evaluation ---


if __name__ == '__main__':
    main()