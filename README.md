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

Матрица гипотез, walk-forward 2017–2026 (2356 дней, издержки 1bp), полная таблица в `artifacts/final_comparison.csv`:

| Стратегия | Sharpe net | Ann ret | Max DD |
|---|---|---|---|
| **SMA200 × vol-forecast (ML-вола)** | **1.01** | **14.2%** | **−19.3%** |
| SMA200 × vol-target (без ML) | 0.99 | 13.5% | −18.0% |
| SMA200 + fixed-horizon meta | 0.98 | 10.5% | −17.4% |
| buy & hold | 0.87 | 16.0% | −33.7% |
| SMA200 + triple-barrier meta | 0.71 | 7.1% | −17.9% |

Ключевые факты: направление SPY почти непредсказуемо (OOS AUC 0.53), волатильность — предсказуема (OOS AUC **0.73** тем же пайплайном). Победитель переживает поправку на множественные тесты: **Deflated Sharpe 98.9%** при 10 испробованных конфигурациях (E[max SR] от бесскилловых попыток = 0.24).


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
