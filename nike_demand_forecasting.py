"""
Nike Demand Forecasting - Predictive Model Implementation
BAN6800 Business Analytics Capstone | Module 4 | William Obubo | Nexford University

This script implements the modelling workflow described in the Module 4 report.
Since Nike's proprietary sell-through data is not publicly available, the workflow
uses a publicly available retail demand dataset (Kaggle: Store Item Demand Forecasting)
as a structural stand-in. The dataset has the same key properties as the Nike pipeline
output: weekly SKU-level sales by store (analogous to channel/region), spanning
multiple years with seasonal patterns. The modelling approach, split strategy,
hyperparameter grids, and evaluation metrics are identical to those described in the report.

Dataset: https://www.kaggle.com/datasets/c2f2783e37e955a7f35b7dfcc20a96ae7ed55ce5/store-item-demand-forecasting-challenge
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ── RANDOM SEED FOR REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING AND INSPECTION
# Mirrors Module 3 pipeline output: NIKE_DEMAND_ANALYTICS_FACT
# Each row = one SKU (item) × one store (channel/region) × one week
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the retail demand dataset and parse dates."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows | Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store'].nunique()} | Items (SKUs): {df['item'].nunique()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# Implements the feature set described in Module 4, Section 2:
# - Lag features (lag-1 through lag-12 weeks)
# - Rolling window averages (4, 8, 12 weeks)
# - Seasonal decomposition index (week-of-year)
# - Temporal features (month, quarter, year)
# Leakage prevention: all lag/rolling features shift data BACKWARD in time
# so no future values appear as features at any observation.
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build analysis-ready feature set per SKU-store (channel) series.
    All lag and rolling operations are applied within each SKU-store group
    to prevent cross-series contamination.
    """
    df = df.sort_values(["store", "item", "date"]).copy()

    # Temporal features
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["year"]         = df["date"].dt.year

    # Lag features (within each SKU-store group to prevent leakage)
    grp = df.groupby(["store", "item"])["sales"]
    for lag in [1, 2, 4, 8, 12, 52]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling window averages (shift by 1 first so current week is not included)
    shifted = grp.shift(1)
    for window in [4, 8, 12]:
        df[f"rolling_mean_{window}w"] = (
            shifted.transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Seasonal index: mean sales for this week-of-year across the training history
    # (computed later, after train/test split, to prevent leakage)
    # Placeholder column created here; filled in Section 3.
    df["seasonal_index"] = np.nan

    # One-hot encode store and item (analogous to channel and product category)
    df = pd.get_dummies(df, columns=["store", "item"], drop_first=True)

    # Drop rows with NaN from lag creation (first 52 weeks per series)
    df = df.dropna(subset=[c for c in df.columns if c.startswith("lag_")])

    print(f"Feature matrix shape after engineering: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TIME-AWARE TRAIN / VALIDATION / TEST SPLIT
# Implements the split strategy from Module 4, Section 3.
# NO random shuffling — future data must never appear in training.
# Split is on calendar date, not row index.
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Expanding window time-aware split.
    train_end : last date in training set  (e.g. '2016-12-31')
    val_end   : last date in validation set (e.g. '2017-06-30')
    Anything after val_end is the held-out test set.
    """
    train = df[df["date"] <= train_end]
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test  = df[df["date"] > val_end]

    # Fill seasonal index using TRAINING data only (leakage prevention)
    seasonal = (
        train.groupby("week_of_year")["sales"]
             .mean()
             .rename("seasonal_index_fill")
    )
    for split in [train, val, test]:
        split["seasonal_index"] = split["week_of_year"].map(seasonal)

    print(f"Train: {len(train):,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Val  : {len(val):,}   rows ({val['date'].min().date()}  to {val['date'].max().date()})")
    print(f"Test : {len(test):,}  rows ({test['date'].min().date()}  to {test['date'].max().date()})")
    return train, val, test


def get_xy(split: pd.DataFrame, target: str = "sales"):
    """Separate features and target; drop date column."""
    drop_cols = [target, "date"]
    X = split.drop(columns=[c for c in drop_cols if c in split.columns])
    y = split[target]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: BASELINE — NAIVE SEASONAL FORECAST
# Prediction = sales in the equivalent week of the prior year (lag-52).
# No model fitting required. Anchors all improvement claims.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(test: pd.DataFrame) -> dict:
    """Naive seasonal baseline: use lag-52 as the prediction."""
    valid = test.dropna(subset=["lag_52"])
    y_true = valid["sales"]
    y_pred = valid["lag_52"]
    return compute_metrics(y_true, y_pred, "Naive Seasonal Baseline")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MODEL TRAINING AND HYPERPARAMETER TUNING
# Three algorithms from Module 4, Section 4:
#   1. Ridge Regression   (regularised linear comparator)
#   2. Random Forest      (ensemble averaging comparator)
#   3. XGBoost            (gradient boosting, primary candidate)
# Tuning uses TimeSeriesSplit cross-validation (5 folds) within training only.
# ─────────────────────────────────────────────────────────────────────────────

def tune_and_train(X_train, y_train, model_name: str):
    """
    Tune hyperparameters using 5-fold TimeSeriesSplit,
    then refit on the full training set with the best parameters.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    if model_name == "ridge":
        model  = Ridge()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
        search = RandomizedSearchCV(
            model, param_grid, cv=tscv, scoring="neg_root_mean_squared_error",
            n_iter=5, random_state=SEED, n_jobs=-1
        )
        search.fit(X_scaled, y_train)
        best = Ridge(**search.best_params_)
        best.fit(X_scaled, y_train)
        print(f"  Ridge best alpha: {search.best_params_['alpha']}")
        return best, scaler

    elif model_name == "random_forest":
        param_dist = {
            "n_estimators": [100, 300, 500],
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_leaf": [1, 2, 4]
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=SEED),
            param_dist, cv=tscv, scoring="neg_root_mean_squared_error",
            n_iter=9, random_state=SEED, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = RandomForestRegressor(**search.best_params_, random_state=SEED)
        best.fit(X_train, y_train)
        print(f"  RF best params: {search.best_params_}")
        return best, None

    elif model_name == "xgboost":
        param_dist = {
            "n_estimators":     [100, 300, 500],
            "max_depth":        [3, 5, 7],
            "learning_rate":    [0.01, 0.05, 0.1],
            "subsample":        [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        search = RandomizedSearchCV(
            XGBRegressor(random_state=SEED, verbosity=0),
            param_dist, cv=tscv, scoring="neg_root_mean_squared_error",
            n_iter=20, random_state=SEED, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = XGBRegressor(**search.best_params_, random_state=SEED, verbosity=0)
        best.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        print(f"  XGBoost best params: {search.best_params_}")
        return best, None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE EVALUATION
# Metrics from Module 4, Section 5:
#   MAE, RMSE, R², MAPE
# All evaluated on the held-out test set only.
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error. Avoids division by zero."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred, label: str) -> dict:
    """Compute and print all four evaluation metrics."""
    mae_val  = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_val   = r2_score(y_true, y_pred)
    mape_val = mape(y_true.values, y_pred if isinstance(y_pred, np.ndarray) else y_pred.values)
    print(f"\n{label}")
    print(f"  MAE  : {mae_val:.2f}")
    print(f"  RMSE : {rmse_val:.2f}")
    print(f"  R²   : {r2_val:.4f}")
    print(f"  MAPE : {mape_val:.2f}%")
    return {"model": label, "MAE": mae_val, "RMSE": rmse_val, "R2": r2_val, "MAPE": mape_val}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FEATURE IMPORTANCE (PERMUTATION)
# Implements the permutation importance analysis from Module 4, Section 6.
# Uses the held-out test set so importance reflects out-of-sample behaviour.
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(model, X_test, y_test, top_n: int = 10):
    """
    Permutation importance on the test set.
    Each feature is shuffled in turn; the drop in R² indicates importance.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=SEED, scoring="r2", n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": result.importances_mean,
        "std":        result.importances_std
    }).sort_values("importance", ascending=False).head(top_n)

    print(f"\nTop {top_n} features by permutation importance (XGBoost, test set):")
    print(imp_df.to_string(index=False))
    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 1: LOAD ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Load data")
    print("=" * 60)
    # Update this path to wherever you saved the Kaggle dataset CSV
    FILEPATH = "train.csv"
    df = load_data(FILEPATH)

    # ── STEP 2: FEATURE ENGINEERING ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering")
    print("=" * 60)
    df = engineer_features(df)

    # ── STEP 3: TIME-AWARE SPLIT ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Time-aware train / validation / test split")
    print("=" * 60)
    # Dataset runs 2013-2017; use 2013-2015 train, 2016 H1 val, 2016 H2+ test
    train, val, test = time_split(df, train_end="2015-12-31", val_end="2016-06-30")
    X_train, y_train = get_xy(train)
    X_val,   y_val   = get_xy(val)
    X_test,  y_test  = get_xy(test)

    # ── STEP 4: BASELINE ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Naive seasonal baseline")
    print("=" * 60)
    baseline_metrics = evaluate_baseline(test)

    # ── STEP 5: TRAIN MODELS ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Model training and hyperparameter tuning")
    print("=" * 60)

    print("\nTraining Ridge Regression...")
    ridge_model, scaler = tune_and_train(X_train, y_train, "ridge")

    print("\nTraining Random Forest...")
    rf_model, _ = tune_and_train(X_train, y_train, "random_forest")

    print("\nTraining XGBoost...")
    xgb_model, _ = tune_and_train(X_train, y_train, "xgboost")

    # ── STEP 6: EVALUATE ON HELD-OUT TEST SET ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Performance evaluation on held-out test set")
    print("=" * 60)

    X_test_scaled = scaler.transform(X_test)
    results = [
        baseline_metrics,
        compute_metrics(y_test, ridge_model.predict(X_test_scaled), "Ridge Regression"),
        compute_metrics(y_test, rf_model.predict(X_test),           "Random Forest"),
        compute_metrics(y_test, xgb_model.predict(X_test),          "XGBoost (selected)"),
    ]

    print("\n--- Summary Table ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # ── STEP 7: FEATURE IMPORTANCE ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Feature importance (XGBoost, permutation analysis)")
    print("=" * 60)
    imp_df = feature_importance(xgb_model, X_test, y_test, top_n=10)

    print("\n" + "=" * 60)
    print("Complete. Results above correspond to the workflow")
    print("documented in the Module 4 report.")
    print("=" * 60)
