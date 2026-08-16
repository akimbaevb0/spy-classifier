"""Meta-labeling pipeline:

primary (Close > SMA200, long/flat) -> ML predicts P(this trade works)
-> confidence threshold filters weak entries.

Threshold and hyperparameters are chosen on the dev segment only (purged
CV, out-of-fold probabilities); evaluation is the same walk-forward
2017+ with monthly refits and costs. Compares buy&hold vs primary alone
vs primary+meta on the identical OOS period.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from config import HORIZON, MIN_TRAIN, REFIT_STEP, SEED
from cv import PurgedKFold
from data import fetch_all, make_label
from evaluate import backtest_positions
from explain import select_features
from features import build_features, transform
from models import fit_ensemble, predict_proba
from primary import primary_signal
from tune import tune

ARTIFACTS = Path(__file__).parent / "artifacts"
THR_GRID = np.arange(0.40, 0.61, 0.01)
MIN_COVERAGE = 0.30  # meta must keep at least 30% of primary's active days


def oof_proba(X, y, w, params, cv) -> pd.Series:
    """Out-of-fold probabilities on the dev segment (for threshold picking)."""
    out = pd.Series(np.nan, index=X.index)
    for tr, te in cv.split(X):
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        out.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


def pick_threshold(oof: pd.Series, fwd: pd.Series) -> float:
    """Max dev Sharpe of daily-ized forward returns among trades kept."""
    best_thr, best_score = 0.5, -np.inf
    daily = (fwd.reindex(oof.index) / HORIZON).dropna()
    oof = oof.reindex(daily.index)
    for thr in THR_GRID:
        kept = daily[oof > thr]
        if len(kept) < len(daily) * MIN_COVERAGE:
            continue
        score = kept.mean() / kept.std() * np.sqrt(252) * len(kept) / len(daily)
        if score > best_score:
            best_thr, best_score = float(thr), score
    return best_thr


def walk_forward_meta(X, y, w, active: pd.Series, params, keep) -> pd.Series:
    """Same engine as run.py, but each refit trains only on days where the
    primary was in the market (that's the population the meta-model serves)."""
    proba = pd.Series(np.nan, index=X.index, name="meta_proba")
    for t0 in range(MIN_TRAIN, len(X), REFIT_STEP):
        tr_end = t0 - HORIZON
        act = active.iloc[:tr_end] == 1
        models = fit_ensemble(X[keep].iloc[:tr_end][act], y.iloc[:tr_end][act],
                              w.iloc[:tr_end][act], params)
        te = slice(t0, min(t0 + REFIT_STEP, len(X)))
        proba.iloc[te] = predict_proba(models, X[keep].iloc[te])
    return proba.dropna()


def main():
    # ---- 1. data, features, label, primary ----
    d = fetch_all()
    spy = d["spy"]
    y_all, w_all = make_label(spy)
    X_all = transform(build_features(d))
    prim_all = primary_signal(spy)
    fwd_all = spy["Close"].shift(-HORIZON) / spy["Close"] - 1

    mask = X_all.notna().all(axis=1) & y_all.notna()
    X, y, w = X_all[mask], y_all[mask].astype(int), w_all[mask]
    prim, fwd = prim_all[mask], fwd_all[mask]
    print(f"samples: {len(X)}, primary in market: {prim.mean():.2%} of days, "
          f"P(win | primary long): {y[prim == 1].mean():.3f}")

    # ---- 2. dev segment: tune + select + threshold on ACTIVE days only ----
    act_dev = (prim.iloc[:MIN_TRAIN] == 1).values
    Xd = X.iloc[:MIN_TRAIN][act_dev]
    yd, wd = y.iloc[:MIN_TRAIN][act_dev], w.iloc[:MIN_TRAIN][act_dev]
    cv = PurgedKFold()

    params, dev_auc = tune(Xd, yd, wd, cv)
    print(f"dev CV AUC (meta): {dev_auc:.4f}")
    keep, report = select_features(Xd, yd, wd, params, cv)
    print(f"selected {len(keep)}/{X.shape[1]} features")

    thr = pick_threshold(oof_proba(Xd[keep], yd, wd, params, cv), fwd)
    print(f"confidence threshold (dev): {thr:.2f}")

    # ---- 3. walk-forward OOS ----
    proba = walk_forward_meta(X, y, w, prim, params, keep)
    idx = proba.index
    act_oos = prim.reindex(idx) == 1
    auc = roc_auc_score(y.reindex(idx)[act_oos], proba[act_oos])
    print(f"\nOOS meta AUC on active days: {auc:.4f} "
          f"({act_oos.sum()} of {len(idx)} days)")

    # positions: decision at t applies to t -> t+1; meta view is 5d,
    # so smooth the binary decision over the horizon to damp turnover
    pos_primary = prim.reindex(idx).astype(float)
    pos_meta = (pos_primary * (proba > thr)).rolling(HORIZON).mean()
    pos_bh = pd.Series(1.0, index=idx)

    print(f"\n===== WALK-FORWARD OOS {idx[0].date()} .. {idx[-1].date()} "
          f"({len(idx)} days) =====")
    rows = {name: backtest_positions(pos, spy) for name, pos in
            [("buy&hold", pos_bh), ("primary (SMA200)", pos_primary),
             ("primary+meta", pos_meta)]}
    print(pd.DataFrame(rows).T.to_string())

    ARTIFACTS.mkdir(exist_ok=True)
    report.to_csv(ARTIFACTS / "meta_importance.csv")
    proba.to_frame().assign(primary=prim.reindex(idx),
                            pos_meta=pos_meta).to_csv(ARTIFACTS / "meta_oos.csv")
    pd.DataFrame(rows).T.to_csv(ARTIFACTS / "meta_metrics.csv")
    print(f"\nartifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
