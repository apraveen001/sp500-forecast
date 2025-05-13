import numpy as np
import matplotlib.pyplot as plt


def generate_signals(predicted_prices: np.ndarray) -> np.ndarray:
    """
    Generate long (1) or flat (0) signals based on predicted price movements.
    Signal[t] = 1 if predicted_prices[t+1] > predicted_prices[t], else 0.
    """
    # compute predicted returns
    returns = (predicted_prices[1:] - predicted_prices[:-1]) / predicted_prices[:-1]
    signals = (returns > 0).astype(int)
    return signals


def compute_strategy_returns(signals: np.ndarray, actual_prices: np.ndarray) -> np.ndarray:
    """
    Compute per-trade returns: if signal=1, capture the actual next-day return; else 0.
    """
    actual_returns = (actual_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]
    # align lengths: signals and actual_returns are same length
    strat_returns = signals * actual_returns
    return strat_returns


def evaluate_strategy(predicted_prices: np.ndarray, actual_prices: np.ndarray) -> dict:
    """
    Evaluate trading strategy performance.
    Returns:
      - signals: array of 0/1 signals
      - strategy_returns: per-trade returns
      - cumulative_returns: cumulative return series
      - avg_return_per_trade: mean return on trades taken
      - total_return: overall return
    """
    signals = generate_signals(predicted_prices)
    strat_returns = compute_strategy_returns(signals, actual_prices)
    # cumulative product of (1 + return)
    cumulative_returns = np.cumprod(1 + strat_returns) - 1
    # avoid division by zero if no trades
    if signals.sum() > 0:
        avg_return_per_trade = strat_returns.sum() / signals.sum()
    else:
        avg_return_per_trade = 0.0
    total_return = cumulative_returns[-1] if cumulative_returns.size > 0 else 0.0

    return {
        'signals': signals,
        'strategy_returns': strat_returns,
        'cumulative_returns': cumulative_returns,
        'avg_return_per_trade': avg_return_per_trade,
        'total_return': total_return
    }


def plot_cumulative_returns(cumulative_returns: np.ndarray, save_path: str = None):
    """
    Plot and optionally save the cumulative return curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns, label='Cumulative Return')
    plt.title('Trading Strategy Cumulative Returns')
    plt.xlabel('Trade Index')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
