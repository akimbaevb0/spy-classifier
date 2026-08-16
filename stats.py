"""Backtest statistics beyond raw Sharpe (AFML ch.14, Bailey & Lopez de Prado).

PSR: probability that the true Sharpe exceeds a benchmark, adjusting for
sample length, skew and kurtosis of returns.
DSR: PSR against the Sharpe one would expect from the BEST of M unskilled
trials — the multiple-testing haircut for "we tried M configurations".
"""
import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649
ANN = np.sqrt(252)


def probabilistic_sharpe(returns, sr_benchmark_daily: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sr = r.mean() / r.std()                       # daily SR
    g3 = sps.skew(r)
    g4 = sps.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    z = (sr - sr_benchmark_daily) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def expected_max_sharpe(trial_sharpes_daily) -> float:
    """E[max SR] across M unskilled trials with the observed SR dispersion."""
    t = np.asarray(trial_sharpes_daily, dtype=float)
    m = len(t)
    if m < 2:
        return 0.0
    sd = t.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1 / m)
                       + EULER_GAMMA * sps.norm.ppf(1 - 1 / (m * np.e))))


def deflated_sharpe(returns, trial_sharpes_annual) -> dict:
    """DSR of `returns` given the annualized Sharpes of ALL trials evaluated."""
    trials_daily = np.asarray(trial_sharpes_annual, dtype=float) / ANN
    sr0 = expected_max_sharpe(trials_daily)
    return {
        "psr_vs_zero": round(probabilistic_sharpe(returns, 0.0), 4),
        "expected_max_sr_ann": round(sr0 * ANN, 2),
        "dsr": round(probabilistic_sharpe(returns, sr0), 4),
        "n_trials": len(trials_daily),
    }
