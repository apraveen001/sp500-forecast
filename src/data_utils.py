# src/data_utils.py

import pandas as pd

def load_sp500_csv(
    path: str,
    skiprows: int = 3,
    date_col: str = "Date",
    cols: list = None
) -> pd.DataFrame:
    """
    Loads the SP500 CSV, skips metadata rows, parses dates,
    and sets the index to the date column.
    """
    if cols is None:
        cols = ["Date", "Close", "High", "Low", "Open", "Volume"]

    df = pd.read_csv(
        path,
        skiprows=skiprows,
        names=cols,
        parse_dates=[date_col],
        index_col=date_col
    )
    return df

def preprocess_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values with forward-fill,
    ensures numeric types,
    and returns cleaned DataFrame.
    """
    # forward-fill NaNs
    df_clean = df.ffill().bfill()
    # enforce numeric dtype on price columns
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    # drop any remaining NaNs
    df_clean = df_clean.dropna()
    return df_clean

if __name__ == "__main__":
    # quick sanity check
    import os
    path = os.path.join(os.path.dirname(__file__), "../data/sp500_20_years.csv")
    df = load_sp500_csv(path)
    df = preprocess_prices(df)
    print("Loaded data:", df.shape)
    print(df.head())
    print("Preprocessed data:", df.shape)
    print(df.head())
    print("Data types:", df.dtypes)
    print("Missing values:", df.isnull().sum().sum())
    print("Data loaded and preprocessed successfully.")
    print("Data types:", df.dtypes)
    print("Missing values:", df.isnull().sum().sum())
    print("Data loaded and preprocessed successfully.")