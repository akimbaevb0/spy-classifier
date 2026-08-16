"""End-to-end: data -> features -> tune (Optuna, purged CV) -> XAI feature
selection (SHAP + MDA) -> retrain -> walk-forward holdout evaluation."""
import numpy as np
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from cv import PurgedKFold
from data import LABEL_YEARS, fetch_spy, make_label
from features import build_features, transform

SEED = 42
HOLDOUT_FRAC = 0.2  # final walk-forward test: most recent 20% of days
N_TRIALS = 60
optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_auc(X, y, params, cv) -> float:
    aucs = []
    for tr, te in cv.split(X):
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def objective_factory(X, y, cv):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 8),      # tiny trees: ~400 samples
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        return cv_auc(X, y, params, cv)
    return objective


def mda_importance(X, y, params, cv, n_rep=10, rng=None) -> pd.Series:
    """Mean Decrease Accuracy: drop in out-of-fold AUC when a feature is shuffled."""
    rng = rng or np.random.default_rng(SEED)
    drops = pd.Series(0.0, index=X.columns)
    for tr, te in cv.split(X):
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        base = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
        for col in X.columns:
            deltas = []
            for _ in range(n_rep):
                Xp = X.iloc[te].copy()
                Xp[col] = rng.permutation(Xp[col].values)
                deltas.append(base - roc_auc_score(y.iloc[te], m.predict_proba(Xp)[:, 1]))
            drops[col] += np.mean(deltas)
    return drops / cv.get_n_splits()


def main():
    # ---- 1. data + label ----
    df = fetch_spy()
    y_all = make_label(df)
    X_all = transform(build_features(df))

    cutoff = df.index[-1] - pd.DateOffset(years=LABEL_YEARS)
    mask = (X_all.index >= cutoff) & X_all.notna().all(axis=1) & y_all.notna()
    X, y = X_all[mask], y_all[mask]
    X, y = X.iloc[:-1], y.iloc[:-1]  # last row's label needs tomorrow's close
    print(f"samples: {len(X)}, features: {X.shape[1]}, "
          f"class balance (up days): {y.mean():.3f}")

    # ---- 2. walk-forward holdout split ----
    n_test = int(len(X) * HOLDOUT_FRAC)
    X_tr, y_tr = X.iloc[:-n_test], y.iloc[:-n_test]
    X_te, y_te = X.iloc[-n_test:], y.iloc[-n_test:]
    cv = PurgedKFold(n_splits=5, embargo=5)

    # ---- 3. baseline: logistic regression ----
    lr_aucs = []
    for tr, te in cv.split(X_tr):
        lr = LogisticRegression(C=0.1, max_iter=1000)
        lr.fit(X_tr.iloc[tr], y_tr.iloc[tr])
        lr_aucs.append(roc_auc_score(y_tr.iloc[te], lr.predict_proba(X_tr.iloc[te])[:, 1]))
    print(f"baseline LogReg CV AUC: {np.mean(lr_aucs):.4f}")

    # ---- 4. Optuna tuning (purged CV inside) ----
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_factory(X_tr, y_tr, cv), n_trials=N_TRIALS)
    best = study.best_params | {"subsample_freq": 1}
    print(f"tuned LGBM CV AUC: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    # ---- 5. XAI feature selection: SHAP + MDA must both vote yes ----
    m_full = LGBMClassifier(**best, random_state=SEED, verbose=-1).fit(X_tr, y_tr)
    shap_vals = shap.TreeExplainer(m_full).shap_values(X_tr)
    if isinstance(shap_vals, list):  # older shap returns [class0, class1]
        shap_vals = shap_vals[1]
    shap_imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_tr.columns)
    mda = mda_importance(X_tr, y_tr, best, cv)

    report = pd.DataFrame({"shap": shap_imp, "mda_auc_drop": mda}).sort_values(
        "shap", ascending=False)
    print("\nimportance report:\n", report.round(4))

    keep = report[(report["shap"] > report["shap"].median()) |
                  (report["mda_auc_drop"] > 0)].index.tolist()
    print(f"\nselected {len(keep)}/{X.shape[1]} features: {keep}")

    sel_auc = cv_auc(X_tr[keep], y_tr, best, cv)
    print(f"CV AUC after selection: {sel_auc:.4f}")
    if sel_auc < study.best_value - 0.005:
        print("selection hurt CV AUC -> keeping full feature set")
        keep = list(X.columns)

    # ---- 6. final out-of-sample evaluation on the untouched holdout ----
    m = LGBMClassifier(**best, random_state=SEED, verbose=-1).fit(X_tr[keep], y_tr)
    proba = m.predict_proba(X_te[keep])[:, 1]
    pred = (proba > 0.5).astype(int)

    print("\n===== HOLDOUT (walk-forward, last "
          f"{n_test} days: {X_te.index[0].date()} .. {X_te.index[-1].date()}) =====")
    print(f"AUC:            {roc_auc_score(y_te, proba):.4f}")
    print(f"accuracy:       {accuracy_score(y_te, pred):.4f}")
    print(f"majority class: {max(y_te.mean(), 1 - y_te.mean()):.4f}  <- must beat this")

    # toy long/flat backtest: long next day when P(up) > 0.5, else flat (no costs)
    fwd_ret = df["Close"].pct_change().shift(-1).reindex(X_te.index)
    strat = fwd_ret * pred
    ann = np.sqrt(252)
    print(f"strategy Sharpe (no costs): {ann * strat.mean() / strat.std():.2f}   "
          f"buy&hold Sharpe: {ann * fwd_ret.mean() / fwd_ret.std():.2f}")


if __name__ == "__main__":
    main()
