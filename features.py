"""Feature engineering + causal transformation.

Rules: every feature is stationary (returns / ratios / oscillators, never
levels except vol indices which get rolling-z-scored anyway) and every
transform uses only past data. Cross-asset series are ffilled onto the SPY
calendar — forward fill only propagates the past, so it is leak-free.
"""
import numpy as np
import pandas as pd

ZSCORE_WIN = 120
CLIP = 3.0


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def _spy_features(spy: pd.DataFrame) -> pd.DataFrame:
    o, c, h, l, v = spy["Open"], spy["Close"], spy["High"], spy["Low"], spy["Volume"]
    r = c.pct_change()

    f = pd.DataFrame(index=spy.index)
    for lag in (1, 2, 3, 5):
        f[f"ret_lag{lag}"] = r.shift(lag - 1)
    for n in (5, 10, 21, 63, 126):
        f[f"mom_{n}"] = c.pct_change(n)
    f["vol_5"] = r.rolling(5).std()
    f["vol_21"] = r.rolling(21).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_21"]
    f["downside_vol_21"] = r.clip(upper=0).rolling(21).std()
    f["rsi_14"] = _rsi(c)
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    f["hl_range"] = (h - l) / c
    f["volume_z"] = (v - v.rolling(21).mean()) / v.rolling(21).std()
    # overnight vs intraday decomposition — different investor populations
    f["overnight_ret"] = o / c.shift(1) - 1
    f["intraday_ret"] = c / o - 1
    return f


def _cross_asset_features(d: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    f = pd.DataFrame(index=index)

    def aligned(name: str) -> pd.Series | None:
        if name not in d:
            return None
        return d[name]["Close"].reindex(index).ffill()

    vix, vix3m, tnx, hyg = (aligned(k) for k in ("vix", "vix3m", "tnx", "hyg"))
    if vix is not None:
        f["vix_level"] = vix
        f["vix_chg_5"] = vix.pct_change(5)
        if vix3m is not None:
            f["vix_term"] = vix3m / vix - 1  # backwardation = stress
    if tnx is not None:
        f["tnx_chg_5"] = tnx.diff(5)
    if hyg is not None:
        f["credit_mom_10"] = hyg.pct_change(10)
        f["credit_vol_21"] = hyg.pct_change().rolling(21).std()
    return f


def build_features(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = d["spy"]
    return pd.concat([_spy_features(spy),
                      _cross_asset_features(d, spy.index)], axis=1)


def transform(f: pd.DataFrame) -> pd.DataFrame:
    """Rolling z-score per feature (past window only), clipped at +-3 sigma."""
    mu = f.rolling(ZSCORE_WIN).mean()
    sd = f.rolling(ZSCORE_WIN).std()
    return ((f - mu) / sd).clip(-CLIP, CLIP)
