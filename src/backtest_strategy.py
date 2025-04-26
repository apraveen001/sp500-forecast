import numpy as np
import pandas as pd


def backtest_strategy(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.0,
    txn_cost: float = 0.0,
    holding_period: int = 1,
    long_short: bool = False
) -> pd.Series:
    """
    Simple backtest:
      - Enter long when pred_return > threshold
      - Enter short when long_short and pred_return < -threshold
      - Hold position for `holding_period` days
      - Subtract txn_cost on each round-trip

    Returns:
        cumulative_returns: pd.Series indexed by entry date
    """
    n = len(prices)
    # expected preds is next-day price prediction
    pred_ret = (preds[1:] - preds[:-1]) / preds[:-1]
    dates_ret = dates[1:]

    returns = []
    entry_dates = []
    for i, r in enumerate(pred_ret[:-holding_period]):
        action = 0
        if r > threshold:
            action = 1
        elif long_short and r < -threshold:
            action = -1
        if action == 0:
            continue
        entry_price = prices[i]
        exit_price = prices[i + holding_period]
        ret = action * (exit_price - entry_price) / entry_price
        # subtract round-trip cost
        ret -= txn_cost
        returns.append(ret)
        entry_dates.append(dates_ret[i])

    cum = pd.Series(returns, index=entry_dates).cumsum()
    cum.index.name = 'EntryDate'
    cum.name = 'CumulativeReturn'
    return cum


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
