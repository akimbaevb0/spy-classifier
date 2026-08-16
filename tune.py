"""Optuna (TPE) over a heavily regularized LGBM space, scored by mean AUC
across purged folds. Tuning runs ONLY on the pre-walk-forward segment, so
hyperparameters never see the evaluation period."""
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from config import SEED, TUNE_TRIALS
from cv import PurgedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_auc(X: pd.DataFrame, y: pd.Series, w: pd.Series, params: dict,
           cv: PurgedKFold) -> float:
    aucs = []
    for tr, te in cv.split(X):
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def _objective(trial, X, y, w, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 16),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
    }
    return cv_auc(X, y, w, params, cv)


def tune(X: pd.DataFrame, y: pd.Series, w: pd.Series,
         cv: PurgedKFold) -> tuple[dict, float]:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: _objective(t, X, y, w, cv), n_trials=TUNE_TRIALS)
    return study.best_params | {"subsample_freq": 1}, study.best_value
