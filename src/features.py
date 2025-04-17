# src/features.py

import pandas as pd
import numpy as np


def compute_RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Compute Bollinger Bands: rolling mean plus/minus num_std * rolling std.
    Returns a DataFrame with columns ['BB_Middle','BB_Upper','BB_Lower'].
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    return pd.DataFrame({
        'BB_Middle': rolling_mean,
        'BB_Upper': upper,
        'BB_Lower': lower
    })


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 'Close' column, compute a set of technical indicators:
      - Moving Averages: MA_10, MA_50, MA_200
      - RSI
      - Bollinger Bands: BB_Middle, BB_Upper, BB_Lower
      - Log Returns
      - Volatility (rolling std of log returns)

    Returns a DataFrame with these features, indexed the same as df.
    """
    feat = pd.DataFrame(index=df.index)

    # Moving averages
    feat['MA_10'] = df['Close'].rolling(window=10).mean()
    feat['MA_50'] = df['Close'].rolling(window=50).mean()
    feat['MA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    feat['RSI'] = compute_RSI(df['Close'], period=14)

    # Bollinger Bands
    bb = compute_bollinger_bands(df['Close'], window=20, num_std=2)
    feat = feat.join(bb)

    # Log returns
    feat['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility: rolling std of log returns
    feat['Volatility'] = feat['Log_Returns'].rolling(window=10).std()

    # Drop initial NaNs
    return feat.dropna()


if __name__ == '__main__':
    # Quick test
    from data_utils import load_sp500_csv, preprocess_prices
    df0 = load_sp500_csv('../data/sp500_20_years.csv')
    df = preprocess_prices(df0)
    feats = build_technical_features(df)
    print(feats.head())
