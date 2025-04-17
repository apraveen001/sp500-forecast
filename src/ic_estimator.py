# src/ic_estimator.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import entropy


def rolling_initial_conditions(
    prices: pd.Series,
    window: int = 252,
    bins: int = 20
) -> pd.DataFrame:
    """
    For each date t >= window, fits ARMA(1,1) + GARCH(1,1) on prices[t-window:t]
    and computes Shannon entropy of the standardized residuals.

    Returns:
        DataFrame indexed by dates[t], with columns:
          ['phi','theta','omega','alpha','beta','entropy']
    """
    ic_list = []
    dates = prices.index[window:]

    for end_date in dates:
        # slice window of prices
        window_prices = prices.loc[:end_date].iloc[-window:]

        # 1) fit ARMA(1,1) for the mean process
        arma_res = ARIMA(window_prices, order=(1, 0, 1)).fit()
        phi = arma_res.params.get('ar.L1', np.nan)
        theta = arma_res.params.get('ma.L1', np.nan)

        # residuals from ARMA
        resid = arma_res.resid

        # 2) fit GARCH(1,1) on residuals
        garch_res = arch_model(resid, p=1, q=1).fit(disp='off')
        omega = garch_res.params.get('omega', np.nan)
        alpha = garch_res.params.get('alpha[1]', np.nan)
        beta = garch_res.params.get('beta[1]', np.nan)

        # 3) standardized residuals
        std_resid = resid / garch_res.conditional_volatility

        # 4) compute Shannon entropy
        hist, _ = np.histogram(std_resid, bins=bins)
        e = entropy(hist + 1e-8)

        ic_list.append((phi, theta, omega, alpha, beta, e))

    cols = ['phi', 'theta', 'omega', 'alpha', 'beta', 'entropy']
    return pd.DataFrame(ic_list, index=dates, columns=cols)


if __name__ == "__main__":
    # simple test
    from data_utils import load_sp500_csv, preprocess_prices
    df = load_sp500_csv("../data/sp500_20_years.csv")
    df = preprocess_prices(df)
    ic_df = rolling_initial_conditions(df['Close'], window=252)
    print(ic_df.head())