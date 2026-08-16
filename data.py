"""Fetch SPY + cross-asset context tickers, build the label.

Label: sign of the 5-day forward return. Sample weight: |forward return|
in units of trailing volatility, clipped — a +0.05% week and a +3% week are
both "up", but the model should not treat them as equally important.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from config import HORIZON, START

TICKERS = {
    "spy": "SPY",
    "vix": "^VIX",      # implied vol level
    "vix3m": "^VIX3M",  # 3m implied vol -> term structure
    "tnx": "^TNX",      # 10y treasury yield
    "hyg": "HYG",       # high-yield credit
}


def fetch_all(start: str = START) -> dict[str, pd.DataFrame]:
    raw = yf.download(list(TICKERS.values()), start=start, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker")
    if raw.empty:
        raise RuntimeError("yfinance returned no data")
    out = {}
    for name, ticker in TICKERS.items():
        if ticker in raw.columns.get_level_values(0):
            df = raw[ticker].dropna(how="all")
            if not df.empty:
                out[name] = df
    if "spy" not in out:
        raise RuntimeError("SPY data missing")
    missing = set(TICKERS) - set(out)
    if missing:
        print(f"warning: no data for {sorted(missing)}, their features are skipped")
    return out


def make_label(spy: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns (binary label, sample weight) for HORIZON-day forward return."""
    close = spy["Close"]
    fwd = close.shift(-HORIZON) / close - 1
    vol = close.pct_change().rolling(21).std()
    z = fwd / (vol * np.sqrt(HORIZON))
    y = (fwd > 0).astype(int).where(fwd.notna()).rename("y")
    w = z.abs().clip(0, 3).rename("w")
    return y, w
