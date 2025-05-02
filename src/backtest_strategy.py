import numpy as np
import pandas as pd

def backtest_strategy(
    dates: pd.DatetimeIndex, # Should align with prices & preds
    prices: np.ndarray,     # Actual prices
    preds: np.ndarray,      # Predicted prices
    threshold: float = 0.0,
    txn_cost: float = 0.0,
    holding_period: int = 1,
    long_short: bool = False
) -> pd.Series:
    """
    Simple backtest with corrected indexing and geometric compounding.
    Signal based on predicted return: (preds[t+1] - preds[t]) / preds[t]
    Trade executed based on prices on day t+1, held for holding_period days.
    """
    n = len(prices)
    # Ensure preds and prices align. Typically preds might be shorter if predicting next day.
    # Let's assume preds[i] is prediction for day i+1, prices[i] is price for day i.
    # We need predicted returns to generate signals for day i+1 onwards.
    # Align lengths carefully. If preds has predictions for dates[1] to dates[n-1]...
    if len(preds) != n:
         # Adjust based on how preds/prices/dates are aligned from evaluate.py
         # Common: preds = model.predict(X_test) -> corresponds to y_test -> dates
         # preds_orig = inverse_scale(preds) -> aligns with dates
         # y_test_orig = inverse_scale(y_test) -> aligns with dates
         # Let's assume preds_orig and y_test_orig passed in align with dates index
         pass # If they align, no adjustment needed here, but check evaluate.py call

    # Calculate predicted returns: pred_ret[i] is predicted return *for day i+1*
    # Requires preds[i+1] and preds[i] -> valid for i from 0 to len(preds)-2
    # Limit loop range based on available predictions and holding period
    pred_ret = (preds[1:] - preds[:-1]) / preds[:-1] # Length len(preds)-1

    returns = []
    entry_dates_list = [] # Use a list first
    
    if len(pred_ret) > 0:
        print(f"\n--- Predicted Returns Stats for Backtest ---")
        print(f"Min Predicted Return: {np.min(pred_ret):.6f}")
        print(f"Max Predicted Return: {np.max(pred_ret):.6f}")
        print(f"Mean Predicted Return: {np.mean(pred_ret):.6f}")
        print(f"Threshold Used: {threshold}\n")
    else:
        print("Predicted returns array (pred_ret) is empty.")

    # Loop through potential *decision* points (day i)
    # Decision on day i impacts trade entry on day i+1
    # Need prices up to i+1+holding_period for exit
    for i in range(len(pred_ret)): # i corresponds to decision time at end of day i
        trade_signal_day_index = i + 1 # Signal is for day i+1
        entry_day_index = i + 1
        exit_day_index = entry_day_index + holding_period

        # Check if we have prices for entry and exit
        if exit_day_index >= n:
            break # Cannot execute trade; exit price is out of bounds

        r = pred_ret[i] # Predicted return for day i+1
        action = 0
        if r > threshold:
            action = 1
        elif long_short and r < -threshold:
            action = -1

        if action == 0:
            continue

        entry_price = prices[entry_day_index] # Enter using price on day i+1
        exit_price = prices[exit_day_index]   # Exit using price on day i+1+holding_period

        # Handle potential division by zero if entry_price is 0
        if entry_price == 0:
            continue

        # Calculate return for this single trade
        trade_ret = action * (exit_price - entry_price) / entry_price
        trade_ret -= txn_cost # Subtract round-trip cost

        returns.append(trade_ret)
        entry_dates_list.append(dates[entry_day_index]) # Record entry date

    if not returns:
        print("Warning: No trades executed in backtest_strategy.")
        return pd.Series(dtype=float) # Return empty series if no trades

    # Use geometric compounding
    entry_dates_pd = pd.DatetimeIndex(entry_dates_list)
    strategy_returns_series = pd.Series(returns, index=entry_dates_pd)
    cumulative_returns = (1 + strategy_returns_series).cumprod() - 1

    cumulative_returns.index.name = 'EntryDate'
    cumulative_returns.name = 'CumulativeReturn'
    return cumulative_returns




if __name__ == '__main__':
    # Quick sanity test
    import matplotlib.pyplot as plt
    dates = pd.date_range('2020-01-01', periods=100)
    prices = np.cumsum(np.random.randn(100) * 0.01) + 100
    preds = prices + np.random.randn(100) * 0.005
    cum = backtest_strategy(dates, prices, preds,
                            threshold=0.001,
                            txn_cost=0.0005,
                            holding_period=3,
                            long_short=True)
    cum.plot(title='Strategy Cumulative Return')
    plt.show()
