# Nike Demand Forecasting — Predictive Model Implementation
**BAN6800 Business Analytics Capstone | Module 4 | William Obubo | Nexford University**

This repository contains the Python implementation of the predictive modelling workflow documented in the Module 4 report: *Demand Forecasting Model for Nike, Inc.: Retail Vertical.*

---

## What This Script Does

Implements all seven steps of the modelling workflow described in the report:

1. Data loading and inspection
2. Feature engineering (lag features, rolling averages, seasonal index)
3. Time-aware train / validation / test split (no data leakage)
4. Naive seasonal baseline (anchor for improvement measurement)
5. Model training and hyperparameter tuning — three algorithms:
   - Ridge Regression (regularised linear comparator)
   - Random Forest (ensemble comparator)
   - XGBoost (primary candidate, gradient boosting)
6. Performance evaluation on the held-out test set (MAE, RMSE, R², MAPE)
7. Feature importance via permutation analysis (XGBoost)

---

## Dataset

Since Nike's proprietary sell-through data is not publicly available, the script uses the **Kaggle Store Item Demand Forecasting** dataset as a structural stand-in. It shares the same key properties as the Nike pipeline output: weekly SKU-level sales by store/channel across multiple years with clear seasonal patterns.

**Download:** https://www.kaggle.com/datasets/c2f2783e37e955a7f35b7dfcc20a96ae7ed55ce5/store-item-demand-forecasting-challenge

Download `train.csv` and place it in the same directory as the script.

---

## Requirements

```
pip install xgboost scikit-learn pandas numpy
```

---

## How to Run

```bash
python nike_demand_forecasting.py
```

Expected output includes training logs, a summary metrics table, and the top 10 features by permutation importance.

---

## File Structure

```
nike-demand-forecasting-ban-capstone/
├── nike_demand_forecasting.py   # Main modelling script
├── README.md                    # This file
```

---

## Methodology Notes

- All lag and rolling features are computed using only past data at each observation, preventing temporal leakage
- Scaler parameters (for Ridge) are fitted on training data only and applied to validation and test sets separately
- Hyperparameter tuning uses 5-fold TimeSeriesSplit cross-validation within the training window only
- The held-out test set is never used during training or tuning

---

*Submitted in response to Module 4 assignment feedback from Dr. Wanjiku, BAN6800 Business Analytics Capstone, Nexford University.*
