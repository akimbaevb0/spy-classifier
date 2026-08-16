"""XAI feature selection: SHAP (what the model uses) x MDA (what actually
predicts out-of-fold). High SHAP + negative MDA = overfitting signature."""
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from config import SEED
from cv import PurgedKFold


def shap_importance(X: pd.DataFrame, y: pd.Series, w: pd.Series,
                    params: dict) -> pd.Series:
    m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
    m.fit(X, y, sample_weight=w)
    vals = shap.TreeExplainer(m).shap_values(X)
    if isinstance(vals, list):  # older shap returns [class0, class1]
        vals = vals[1]
    return pd.Series(np.abs(vals).mean(axis=0), index=X.columns)


def mda_importance(X: pd.DataFrame, y: pd.Series, w: pd.Series, params: dict,
                   cv: PurgedKFold, n_rep: int = 10) -> pd.Series:
    """Mean Decrease Accuracy: out-of-fold AUC drop when a feature is shuffled."""
    rng = np.random.default_rng(SEED)
    drops = pd.Series(0.0, index=X.columns)
    for tr, te in cv.split(X):
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr])
        base = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
        for col in X.columns:
            deltas = []
            for _ in range(n_rep):
                Xp = X.iloc[te].copy()
                Xp[col] = rng.permutation(Xp[col].values)
                deltas.append(base - roc_auc_score(y.iloc[te], m.predict_proba(Xp)[:, 1]))
            drops[col] += np.mean(deltas)
    return drops / cv.get_n_splits()


def select_features(X: pd.DataFrame, y: pd.Series, w: pd.Series, params: dict,
                    cv: PurgedKFold) -> tuple[list[str], pd.DataFrame]:
    report = pd.DataFrame({
        "shap": shap_importance(X, y, w, params),
        "mda_auc_drop": mda_importance(X, y, w, params, cv),
    }).sort_values("shap", ascending=False)
    keep = report[(report["shap"] > report["shap"].median()) |
                  (report["mda_auc_drop"] > 0)].index.tolist()
    return keep, report
