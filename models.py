"""Seed-ensemble around LightGBM: on noisy tabular data a single seed's tree
structure is close to a lottery draw; averaging probabilities across seeds
removes that variance without touching bias."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import SEEDS


def fit_ensemble(X: pd.DataFrame, y: pd.Series, w: pd.Series,
                 params: dict, seeds=SEEDS) -> list[LGBMClassifier]:
    return [LGBMClassifier(**params, random_state=s, verbose=-1)
            .fit(X, y, sample_weight=w) for s in seeds]


def predict_proba(models: list[LGBMClassifier], X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
