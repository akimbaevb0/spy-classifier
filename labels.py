"""Alternative labels: triple-barrier meta-labels and volatility-expansion.

Triple-barrier (AFML ch.3): for each day, a long trade exits at the first of
profit-take (+pt*sigma), stop-loss (-sl*sigma) or the vertical barrier
(HORIZON days). Meta-label = did the trade make money. Uses High/Low for
touch detection, not just closes. Weight = |exit return| in vol units
(return attribution, AFML ch.5, uniqueness omitted for daily non-overlapping
entry grid).
"""
import numpy as np
import pandas as pd

from config import HORIZON


def triple_barrier_meta(spy: pd.DataFrame, horizon: int = HORIZON,
                        pt: float = 1.0, sl: float = 1.0
                        ) -> tuple[pd.Series, pd.Series]:
    c, h, l = spy["Close"], spy["High"], spy["Low"]
    daily_vol = c.pct_change().ewm(span=21).std()
    tgt = daily_vol * np.sqrt(horizon)  # barrier half-width over the horizon

    y = pd.Series(np.nan, index=c.index, name="y")
    w = pd.Series(np.nan, index=c.index, name="w")
    cv, hv, lv, tv = c.values, h.values, l.values, tgt.values

    for i in range(len(cv) - horizon):
        if not np.isfinite(tv[i]) or tv[i] <= 0:
            continue
        p0 = cv[i]
        up, dn = p0 * (1 + pt * tv[i]), p0 * (1 - sl * tv[i])
        ret, days = None, horizon
        for j in range(i + 1, i + 1 + horizon):
            if lv[j] <= dn:            # stop-loss first (conservative order)
                ret, days = dn / p0 - 1, j - i
                break
            if hv[j] >= up:            # profit-take
                ret, days = up / p0 - 1, j - i
                break
        if ret is None:                # vertical barrier
            ret = cv[i + horizon] / p0 - 1
        y.iloc[i] = float(ret > 0)
        w.iloc[i] = min(abs(ret) / (tv[i] * np.sqrt(days / horizon)), 3.0)
    return y, w


def vol_expansion_label(spy: pd.DataFrame, horizon: int = HORIZON
                        ) -> tuple[pd.Series, pd.Series]:
    """1 if realized vol over the next `horizon` days exceeds trailing 21d vol."""
    r = spy["Close"].pct_change()
    rv_fwd = r.rolling(horizon).std().shift(-horizon)
    rv_trail = r.rolling(21).std()
    y = (rv_fwd > rv_trail).astype(float).where(rv_fwd.notna()).rename("y")
    w = np.log(rv_fwd / rv_trail).abs().clip(0, 3).rename("w")
    return y, w
