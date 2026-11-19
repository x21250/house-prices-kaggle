"""
House Prices - Advanced Regression Techniques
Clean, runnable version of the original `notebooks_Untitled.ipynb`.

What was wrong in the original, and what changed:

  1. Target scale was inconsistent.  `grid_search.fit(X, np.log(y))` trained on a
     logged target, but `predict()` (still in log space) went straight into the
     submission file, and the scorer was `neg_root_mean_squared_log_error`, which
     takes log1p of an already-logged value.  Now: fit on log1p(y), score with
     plain RMSE (identical to RMSLE on that scale), and expm1 before writing.
  2. `OrdinalEncoder` was applied to every categorical, which forces an arbitrary
     order onto `Neighborhood` and hands a linear model numbers where it needs
     indicators.  Now: one-hot for categoricals.
  3. The step holding XGBoost was named `rf`, so every param key read
     `rf__learning_rate` and so on.  Renamed to `model`.
  4. The grid was 432 candidates x 5 folds = 2160 fits.  Trimmed to 18 x 5 = 90;
     the extra cells were re-fitting near-duplicate configurations.
  5. Cell 9 silently overwrote `pipeline` with a LassoCV that was never used, and
     `best_model` still pointed at the grid search.  Dead code removed.
  6. Submission used `id` (lowercase) and `test['Id']`; Kaggle expects `Id`.
  7. `select_dtypes(include=['object'])` returns nothing under pandas 3, where
     strings are dtype `str`.  Column detection now uses dtype kind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_DIR = Path.home() / "Downloads/house-prices-advanced-regression-techniques"
OUT_DIR = Path(__file__).resolve().parent.parent  # repo root
RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_data(data_dir: Path = DATA_DIR):
    """Return (X_train, y_train, X_test, test_ids)."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    y = train.pop("SalePrice")
    train = train.drop(columns=["Id"])
    test_ids = test.pop("Id")
    return train, y, test, test_ids


def split_columns(X: pd.DataFrame):
    """Numeric vs categorical by dtype kind.

    `include=['object']` misses pandas 3's `str` dtype, which would silently send
    every categorical column to the numeric imputer.
    """
    cat = [c for c in X.columns if X[c].dtype.kind in "OUS" or str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return cat, num


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def make_preprocessor(cat_cols, num_cols, scale_numeric: bool = True) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def build_xgb_pipeline(preprocessor) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_lasso_pipeline(preprocessor) -> Pipeline:
    return Pipeline([
        ("prep", preprocessor),
        ("model", LassoCV(cv=5, max_iter=50000, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


XGB_GRID = {
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__n_estimators": [300, 600],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    X, y, X_test, test_ids = load_data()
    y_log = np.log1p(y)
    cat_cols, num_cols = split_columns(X)
    print(f"train {X.shape}  test {X_test.shape}  categorical {len(cat_cols)}  numeric {len(num_cols)}")

    pre = make_preprocessor(cat_cols, num_cols)

    # --- baseline: LassoCV, no tuning -------------------------------------
    lasso = build_lasso_pipeline(make_preprocessor(cat_cols, num_cols))
    lasso_scores = cross_val_score(
        lasso, X, y_log, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"lasso  CV RMSLE {-lasso_scores.mean():.5f} +/- {lasso_scores.std():.5f}")

    # --- XGBoost with a grid ----------------------------------------------
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(
        build_xgb_pipeline(pre),
        param_grid=XGB_GRID,
        cv=CV,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y_log)

    cv_rmse = -grid.best_score_
    print(f"\nbest params: {grid.best_params_}")
    print(f"xgb    CV RMSLE {cv_rmse:.5f}")

    # --- refit on all data and predict ------------------------------------
    best_model = grid.best_estimator_.fit(X, y_log)
    y_pred = np.expm1(best_model.predict(X_test))

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": np.clip(y_pred, 30_000, 800_000),
    })
    submission.to_csv(OUT_DIR / "submission_notebook.csv", index=False)
    print(f"\nwrote submission_notebook.csv  rows={len(submission)}  "
          f"range={submission.SalePrice.min():.0f}-{submission.SalePrice.max():.0f}")

    # round-trip check: are the predictions on the price scale?
    assert 50_000 < submission.SalePrice.median() < 400_000, "predictions look like log values"


if __name__ == "__main__":
    main()
