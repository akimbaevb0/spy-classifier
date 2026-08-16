# SPY Direction Classifier

Research-grade пайплайн классификации направления SPY (5-дневный горизонт):

- **Данные:** 15 лет (2010+), ~3850 сэмплов; SPY + кросс-активный контекст (VIX, VIX3M term structure, 10Y yield, HYG credit)
- **Лейбл:** знак 5-дневного forward return, sample weights = |return| в единицах волатильности
- **Фичи:** 26 стационарных, rolling z-score 120d (без look-ahead), overnight/intraday декомпозиция
- **CV:** Purged K-Fold + embargo (López de Prado)
- **Тюнинг:** Optuna TPE, только на dev-сегменте (первые ~6 лет) — оценочный период не видит гиперпараметры
- **Отбор фичей:** SHAP × MDA (out-of-fold) с требованием согласия
- **Оценка:** walk-forward 2017–2026, expanding window, перефит ежемесячно, seed-ансамбль ×5, издержки 1bp

## Результат (честный)

Walk-forward OOS, 2356 дней: **AUC 0.529**, hit rate 0.57, Sharpe net 0.62 vs buy&hold 0.87.
Слабый положительный ranking-сигнал есть; как standalone long/short стратегия B&H не бьёт.

## Запуск

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python run.py
```

## Структура

| Файл | Что делает |
|------|-----------|
| `config.py` | все параметры пайплайна |
| `data.py` | загрузка тикеров, лейбл + веса |
| `features.py` | SPY + кросс-активные фичи, каузальная трансформация |
| `cv.py` | PurgedKFold с embargo |
| `models.py` | seed-ансамбль LightGBM |
| `tune.py` | Optuna на purged CV |
| `explain.py` | SHAP + MDA отбор |
| `evaluate.py` | walk-forward движок + бэктест с издержками |
| `run.py` | оркестрация, артефакты в `artifacts/` |
