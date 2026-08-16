"""Final hypothesis matrix on one identical walk-forward window (2017+):

  H2  triple-barrier meta-labels vs fixed-horizon (reuses saved fixed-horizon
      meta positions from artifacts/meta_oos.csv for a fair comparison)
  H3  volatility-expansion prediction — verify vol is genuinely predictable
  H4  final strategy: SMA200 primary x TB-meta gate x vol-target sizing
      (H4a: sizing from trailing RV; H4b: RV blended with the H3 forecast)

Every ML layer is tuned on the dev segment only. The comparison table gets
the multiple-testing haircut: PSR + Deflated Sharpe across ALL strategy
configurations evaluated OOS in this project.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import HORIZON, MIN_TRAIN
from cv import PurgedKFold
from data import fetch_all
from evaluate import ANN, backtest_positions
from explain import select_features
from features import build_features, transform
from labels import triple_barrier_meta, vol_expansion_label
from primary import primary_signal
from run_meta import oof_proba, pick_threshold, walk_forward_meta
from stats import deflated_sharpe
from tune import tune

ARTIFACTS = Path(__file__).parent / "artifacts"
TARGET_VOL = 0.15   # annualized vol target for sizing
MAX_LEV = 1.5


def fit_layer(X, y, w, active, cv, label):
    """Tune on dev active days, select features, pick nothing else."""
    act = active.iloc[:MIN_TRAIN] if active is not None else pd.Series(
        True, index=X.index[:MIN_TRAIN])
    m = act.values.astype(bool)
    Xd, yd, wd = X.iloc[:MIN_TRAIN][m], y.iloc[:MIN_TRAIN][m], w.iloc[:MIN_TRAIN][m]
    params, dev_auc = tune(Xd, yd, wd, cv)
    keep, _ = select_features(Xd, yd, wd, params, cv)
    print(f"[{label}] dev CV AUC: {dev_auc:.4f}, features kept: {len(keep)}")
    return params, keep, (Xd, yd, wd)


def main():
    d = fetch_all()
    spy = d["spy"]
    X_all = transform(build_features(d))
    prim_all = primary_signal(spy)
    fwd_all = spy["Close"].shift(-HORIZON) / spy["Close"] - 1

    y_tb_all, w_tb_all = triple_barrier_meta(spy)
    y_v_all, w_v_all = vol_expansion_label(spy)

    mask = (X_all.notna().all(axis=1) & y_tb_all.notna() & y_v_all.notna()
            & w_v_all.notna())
    X = X_all[mask]
    y_tb, w_tb = y_tb_all[mask].astype(int), w_tb_all[mask]
    y_v, w_v = y_v_all[mask].astype(int), w_v_all[mask]
    prim, fwd = prim_all[mask], fwd_all[mask]
    print(f"samples: {len(X)}, TB win rate | primary long: "
          f"{y_tb[prim == 1].mean():.3f}, vol-expansion base rate: {y_v.mean():.3f}")

    cv = PurgedKFold()

    # ---- H2: triple-barrier meta layer ----
    tb_params, tb_keep, (Xd, yd, wd) = fit_layer(X, y_tb, w_tb, prim == 1, cv, "H2 tb-meta")
    tb_thr = pick_threshold(oof_proba(Xd[tb_keep], yd, wd, tb_params, cv), fwd)
    print(f"[H2] threshold (dev): {tb_thr:.2f}")
    proba_tb = walk_forward_meta(X, y_tb, w_tb, prim, tb_params, tb_keep)

    # ---- H3: volatility-expansion layer (trained on all days) ----
    v_params, v_keep, _ = fit_layer(X, y_v, w_v, None, cv, "H3 vol")
    proba_v = walk_forward_meta(X, y_v, w_v, pd.Series(1, index=X.index),
                                v_params, v_keep)

    idx = proba_tb.index.intersection(proba_v.index)
    act = (prim.reindex(idx) == 1)
    auc_tb = roc_auc_score(y_tb.reindex(idx)[act], proba_tb.reindex(idx)[act])
    auc_v = roc_auc_score(y_v.reindex(idx), proba_v.reindex(idx))
    print(f"\nOOS AUC — TB meta (active days): {auc_tb:.4f} | vol expansion: {auc_v:.4f}")

    # ---- positions on the common OOS index ----
    r = spy["Close"].pct_change()
    rv_ann = (r.rolling(21).std() * ANN).reindex(idx)
    lev_a = (TARGET_VOL / rv_ann).clip(0, MAX_LEV)
    rv_blend = rv_ann * (0.75 + 0.5 * proba_v.reindex(idx))   # H3 forecast blend
    lev_b = (TARGET_VOL / rv_blend).clip(0, MAX_LEV)

    p = prim.reindex(idx).astype(float)
    gate_tb = (p * (proba_tb.reindex(idx) > tb_thr)).rolling(HORIZON).mean()

    strategies = {
        "buy&hold": pd.Series(1.0, index=idx),
        "primary (SMA200)": p,
        "primary x vol-target (no ML)": p * lev_a,
        "H2 primary x TB-meta": gate_tb,
        "H4a TB-meta x vol-target": gate_tb * lev_a,
        "H4b TB-meta x vol-forecast": gate_tb * lev_b,
    }
    # fixed-horizon meta from the earlier run, same period, for comparison
    meta_csv = ARTIFACTS / "meta_oos.csv"
    if meta_csv.exists():
        prev = pd.read_csv(meta_csv, index_col=0, parse_dates=True)["pos_meta"]
        strategies["fixed-horizon meta (v3)"] = prev.reindex(idx)

    print(f"\n===== WALK-FORWARD OOS {idx[0].date()} .. {idx[-1].date()} "
          f"({len(idx)} days) =====")
    rows = {name: backtest_positions(pos.fillna(0), spy)
            for name, pos in strategies.items()}
    table = pd.DataFrame(rows).T.sort_values("sharpe_net", ascending=False)
    print(table.to_string())

    # ---- multiple-testing haircut for the winner ----
    # every strategy configuration this project evaluated OOS:
    session_trials = [0.62,   # v2 standalone long/short
                      0.97, 0.98,                     # primary, fixed-horizon meta
                      *[m["sharpe_net"] for m in rows.values()]]
    best_name = table.index[0]
    best_pos = strategies[best_name].fillna(0)
    next_ret = spy["Close"].pct_change().shift(-1).reindex(idx)
    turn = best_pos.diff().abs().fillna(0)
    best_rets = (best_pos * next_ret - turn * 1.0 / 1e4).dropna()
    ds = deflated_sharpe(best_rets, session_trials)
    print(f"\nbest strategy: {best_name}")
    print(f"PSR (true Sharpe > 0):        {ds['psr_vs_zero']:.2%}")
    print(f"E[max Sharpe | {ds['n_trials']} unskilled trials]: "
          f"{ds['expected_max_sr_ann']}")
    print(f"Deflated Sharpe (skill beyond selection bias): {ds['dsr']:.2%}")

    ARTIFACTS.mkdir(exist_ok=True)
    table.to_csv(ARTIFACTS / "final_comparison.csv")
    pd.DataFrame({"proba_tb": proba_tb, "proba_vol": proba_v}).to_csv(
        ARTIFACTS / "final_probas.csv")
    print(f"\nartifacts saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
