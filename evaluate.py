"""Walk-forward evaluation engine + cost-aware backtest.

Expanding window, monthly refits, HORIZON-day purge gap between train end and
prediction start. Positions average the last HORIZON overlapping signals
(each signal is a 5-day view), which also damps turnover.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import COST_BPS, HORIZON, MIN_TRAIN, REFIT_STEP
from models import fit_ensemble, predict_proba

ANN = np.sqrt(252)


def walk_forward(X: pd.DataFrame, y: pd.Series, w: pd.Series,
                 params: dict, keep: list[str]) -> pd.Series:
    proba = pd.Series(np.nan, index=X.index, name="proba")
    for t0 in range(MIN_TRAIN, len(X), REFIT_STEP):
        tr_end = t0 - HORIZON  # purge: last train labels reach into test
        models = fit_ensemble(X[keep].iloc[:tr_end], y.iloc[:tr_end],
                              w.iloc[:tr_end], params)
        te = slice(t0, min(t0 + REFIT_STEP, len(X)))
        proba.iloc[te] = predict_proba(models, X[keep].iloc[te])
    return proba.dropna()


def evaluate(proba: pd.Series, y: pd.Series, spy: pd.DataFrame) -> dict:
    y_te = y.reindex(proba.index)
    auc = roc_auc_score(y_te, proba)
    hit = ((proba > 0.5).astype(int) == y_te).mean()
    majority = max(y_te.mean(), 1 - y_te.mean())

    # long/short backtest: signal in [-1, 1], averaged over overlapping horizon
    sig = (2 * proba - 1).clip(-1, 1)
    pos = sig.rolling(HORIZON).mean()
    next_ret = spy["Close"].pct_change().shift(-1).reindex(pos.index)
    turnover = pos.diff().abs().fillna(0)
    strat = (pos * next_ret - turnover * COST_BPS / 1e4).dropna()
    bh = next_ret.reindex(strat.index)

    def sharpe(r): return float(ANN * r.mean() / r.std())

    def max_dd(r):
        curve = (1 + r).cumprod()
        return float((curve / curve.cummax() - 1).min())

    return {
        "period": f"{proba.index[0].date()} .. {proba.index[-1].date()}",
        "n_days": len(proba),
        "auc": round(auc, 4),
        "hit_rate": round(float(hit), 4),
        "majority_class": round(float(majority), 4),
        "strat_sharpe_net": round(sharpe(strat), 2),
        "buyhold_sharpe": round(sharpe(bh), 2),
        "strat_ann_ret": round(float(strat.mean() * 252), 4),
        "strat_max_dd": round(max_dd(strat), 4),
        "avg_daily_turnover": round(float(turnover.mean()), 4),
    }
