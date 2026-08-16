"""Fetch SPY OHLCV and build the label."""
import pandas as pd
import yfinance as yf

LABEL_YEARS = 2
WARMUP_DAYS = 90  # extra history so rolling features have no NaN inside the label window


def fetch_spy() -> pd.DataFrame:
    # ~2y label window + warmup for the longest rolling lookback (63d) + z-score window
    df = yf.download("SPY", period="3y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data for SPY")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def make_label(df: pd.DataFrame) -> pd.Series:
    """Binary label: 1 if next day's close-to-close return > 0."""
    fwd_ret = df["Close"].pct_change().shift(-1)
    return (fwd_ret > 0).astype(int).rename("y")
