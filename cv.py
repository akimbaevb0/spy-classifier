"""Purged K-Fold with embargo (Lopez de Prado, 'Advances in Financial ML').

The label at t uses prices through t+HORIZON, so train samples whose label
window overlaps the test fold are purged; the embargo drops extra train
samples immediately AFTER the test fold to kill serial-correlation leakage.
"""
import numpy as np

from config import EMBARGO, HORIZON, N_SPLITS


class PurgedKFold:
    def __init__(self, n_splits: int = N_SPLITS, horizon: int = HORIZON,
                 embargo: int = EMBARGO):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        for test_idx in np.array_split(np.arange(n), self.n_splits):
            t0, t1 = test_idx[0], test_idx[-1]
            train_mask = np.ones(n, dtype=bool)
            train_mask[max(0, t0 - self.horizon): t1 + 1] = False  # purge + test
            train_mask[t1 + 1: t1 + 1 + self.embargo] = False      # embargo
            yield np.where(train_mask)[0], test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
