# SPY Direction Classifier

ML-классификатор направления SPY (next-day up/down) с quant-роадмапом:
стационарные фичи + rolling z-score, Purged K-Fold с embargo, Optuna (TPE),
SHAP + MDA для отбора фичей, walk-forward holdout.

## Запуск

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python yfinance pandas numpy scikit-learn lightgbm optuna shap
.venv/bin/python run.py
```

## Файлы

- `data.py` — загрузка SPY (yfinance), бинарный лейбл
- `features.py` — 16 фичей + причинная трансформация (rolling z-score, clip ±3σ)
- `cv.py` — PurgedKFold с embargo (López de Prado)
- `run.py` — baseline → Optuna tuning → SHAP+MDA отбор → holdout-оценка
