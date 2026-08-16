"""Purged K-Fold with embargo (Lopez de Prado, 'Advances in Financial ML').

Label at t uses information through t+1 (next-day return), so train samples
whose label window overlaps the test fold are purged; an embargo drops extra
train samples immediately AFTER the test fold to kill serial-correlation leak.
"""
import numpy as np

LABEL_HORIZON = 1  # our label looks 1 day ahead


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_bounds = np.array_split(np.arange(n), self.n_splits)
        for test_idx in fold_bounds:
            t0, t1 = test_idx[0], test_idx[-1]
            train_mask = np.ones(n, dtype=bool)
            # test fold itself + purge: train labels reaching into the test window
            train_mask[max(0, t0 - LABEL_HORIZON): t1 + 1] = False
            # embargo after the test fold
            train_mask[t1 + 1: t1 + 1 + self.embargo] = False
            yield np.where(train_mask)[0], test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
