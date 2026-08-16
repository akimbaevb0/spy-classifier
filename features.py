"""Feature engineering + causal transformation.

Every feature is stationary (returns/ratios, never price levels) and every
transform uses only past data (rolling windows, shift) — no look-ahead.
"""
import numpy as np
import pandas as pd

ZSCORE_WIN = 60
CLIP = 3.0


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    r = c.pct_change()

    f = pd.DataFrame(index=df.index)
    # lagged returns (short-term reversal signal)
    for lag in (1, 2, 3, 5):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)
    # momentum over several horizons
    for n in (5, 10, 21, 63):
        f[f"mom_{n}"] = c.pct_change(n)
    # volatility regime
    f["vol_5"] = r.rolling(5).std()
    f["vol_21"] = r.rolling(21).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_21"]  # short vs long vol regime
    # oscillator / mean-reversion
    f["rsi_14"] = _rsi(c)
    f["dist_sma20"] = c / c.rolling(20).mean() - 1
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    # intraday range and volume pressure
    f["hl_range"] = (h - l) / c
    f["vol_z"] = (v - v.rolling(21).mean()) / v.rolling(21).std()
    return f


def transform(f: pd.DataFrame) -> pd.DataFrame:
    """Rolling z-score per feature (past window only), clipped at +-3 sigma."""
    mu = f.rolling(ZSCORE_WIN).mean()
    sd = f.rolling(ZSCORE_WIN).std()
    return ((f - mu) / sd).clip(-CLIP, CLIP)
