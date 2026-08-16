"""Single source of truth for pipeline knobs."""
START = "2010-01-01"   # ~15y of daily data -> ~3800 samples
HORIZON = 5            # label looks 5 trading days ahead
EMBARGO = 5            # extra days dropped after each test fold
N_SPLITS = 5
SEEDS = range(5)       # seed-ensemble size (LGBM is unstable on tabular noise)
TUNE_TRIALS = 60
MIN_TRAIN = 1500       # walk-forward evaluation starts after ~6y of history
REFIT_STEP = 21        # refit the ensemble every month
COST_BPS = 1.0         # one-way transaction cost on traded notional
SEED = 42
