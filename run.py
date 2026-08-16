"""End-to-end research pipeline:
data -> features -> tune on early segment (purged CV) -> XAI feature
selection -> walk-forward evaluation with monthly refits and costs.

The evaluation period is never touched by tuning or feature selection.
"""
from pathlib import Path

import pandas as pd

from config import MIN_TRAIN
from cv import PurgedKFold
from data import fetch_all, make_label
from evaluate import evaluate, walk_forward
from explain import select_features
from features import build_features, transform
from tune import cv_auc, tune

ARTIFACTS = Path(__file__).parent / "artifacts"


def main():
    # ---- 1. data + label ----
    d = fetch_all()
    spy = d["spy"]
    y_all, w_all = make_label(spy)
    X_all = transform(build_features(d))

    mask = X_all.notna().all(axis=1) & y_all.notna()
    X, y, w = X_all[mask], y_all[mask].astype(int), w_all[mask]
    print(f"samples: {len(X)}, features: {X.shape[1]}, up days: {y.mean():.3f}")

    if len(X) <= MIN_TRAIN + 250:
        raise RuntimeError(f"not enough history: {len(X)} rows, need > {MIN_TRAIN + 250}")

    # ---- 2. tuning segment = data before the walk-forward start ----
    X_dev, y_dev, w_dev = X.iloc[:MIN_TRAIN], y.iloc[:MIN_TRAIN], w.iloc[:MIN_TRAIN]
    cv = PurgedKFold()
    params, dev_auc = tune(X_dev, y_dev, w_dev, cv)
    print(f"dev CV AUC (tuned): {dev_auc:.4f}")
    print(f"best params: {params}")

    # ---- 3. XAI feature selection on the same dev segment ----
    keep, report = select_features(X_dev, y_dev, w_dev, params, cv)
    print("\nimportance report:\n", report.round(4))
    sel_auc = cv_auc(X_dev[keep], y_dev, w_dev, params, cv)
    print(f"\nselected {len(keep)}/{X.shape[1]} features, dev CV AUC: {sel_auc:.4f}")
    if sel_auc < dev_auc - 0.005:
        print("selection hurt dev AUC -> keeping full feature set")
        keep = list(X.columns)

    # ---- 4. walk-forward out-of-sample evaluation ----
    proba = walk_forward(X, y, w, params, keep)
    metrics = evaluate(proba, y, spy)
    print("\n===== WALK-FORWARD OOS =====")
    for k, v in metrics.items():
        print(f"{k:>20}: {v}")

    # ---- 5. save artifacts ----
    ARTIFACTS.mkdir(exist_ok=True)
    report.to_csv(ARTIFACTS / "importance.csv")
    proba.to_frame().join(y).to_csv(ARTIFACTS / "oos_predictions.csv")
    pd.Series(metrics).to_csv(ARTIFACTS / "metrics.csv")
    print(f"\nartifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
