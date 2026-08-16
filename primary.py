"""Primary strategy for meta-labeling: long-only trend filter.

Long when Close > SMA200, flat otherwise. The ML layer on top does not
predict the market — it predicts whether THIS primary trade will work,
and filters out the weak entries (Lopez de Prado's meta-labeling).
"""
import pandas as pd


def primary_signal(spy: pd.DataFrame) -> pd.Series:
    close = spy["Close"]
    return (close > close.rolling(200).mean()).astype(int).rename("primary")
