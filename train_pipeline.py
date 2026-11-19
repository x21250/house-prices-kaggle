"""
Kaggle: House Prices - Advanced Regression Techniques
Strong, reproducible pipeline: feature engineering + 6 models + OOF stacking.

Run:  .venv/bin/python train_pipeline.py
Out:  submission.csv, cv_report.md
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

def is_string(s: pd.Series) -> bool:
    """pandas 3 stores strings as dtype 'str' (kind O), not 'object'."""
    return pd.api.types.is_string_dtype(s) or s.dtype.name in ("object", "category")


DATA = Path.home() / "Downloads/house-prices-advanced-regression-techniques"
OUT = Path(__file__).resolve().parent
SEED = 42

# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
test_ids = test["Id"].copy()
y = np.log1p(train["SalePrice"].values)
train = train.drop(columns=["SalePrice"])

n_train = len(train)
df = pd.concat([train, test], axis=0, ignore_index=True)

# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
# 2a. NA that means "the house does not have this feature" -> 0
none_as_zero = [
    "GarageYrBlt", "GarageCars", "GarageArea", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]
for c in none_as_zero:
    df[c] = df[c].fillna(0)

# 2b. LotFrontage: houses on the same street share a similar frontage
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda s: s.fillna(s.median())
)
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

# 2c. categorical NA -> "None" (a real category, not missing)
none_as_category = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
for c in none_as_category:
    df[c] = df[c].fillna("None")

# remaining few missing values -> mode
for c in df.columns:
    if is_string(df[c]) and df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

# 2d. ordinal encodings (quality is ordered; one-hot would throw away the order)
ord_maps = {
    "ExterQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
}
for c, m in ord_maps.items():
    if c in df.columns:
        df[c] = df[c].map(m).astype(float)

# 2e. domain features
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"] = (
    df["FullBath"] + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
)
df["TotalPorch"] = (
    df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"]
    + df["ScreenPorch"] + df["WoodDeckSF"]
)
df["Age"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
df["QualSF"] = df["OverallQual"] * df["TotalSF"]
df["ExterScore"] = df["ExterQual"] * df["ExterCond"]
df["GarageScore"] = df["GarageCars"] * df["GarageArea"]
df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"].replace(0, np.nan)
df["BsmtFinRatio"] = df["BsmtFinSF1"] / df["TotalBsmtSF"].replace(0, np.nan)
df["PorchRatio"] = df["TotalPorch"] / df["GrLivArea"]
df["AvgRoomSF"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, np.nan)
df["BsmtScore"] = df["BsmtQual"].fillna(0) * df["TotalBsmtSF"]
df["KitchenScore"] = df["KitchenQual"] * df["KitchenAbvGr"]

# 2f. log-transform skewed numeric features (linear models care a lot)
num_cols = [
    c for c in df.columns
    if df[c].dtype.name in ("int64", "float64", "int32", "float32")
]
skewed = [c for c in num_cols if abs(df[c].skew()) > 0.75]
for c in skewed:
    if df[c].min() >= 0:
        df[c] = np.log1p(df[c])

# 2g. constant columns
df = df.loc[:, df.nunique(dropna=False) > 1]

# ----------------------------------------------------------------------------
# 3. Split back, drop the two well-known outliers
# ----------------------------------------------------------------------------
X_all = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy().reset_index(drop=True)

outlier_mask = (X_all["GrLivArea"] > np.log1p(4000)) & (y < np.log(300000))
print(f"dropping {int(outlier_mask.sum())} outliers")
X_all = X_all.loc[~outlier_mask].reset_index(drop=True)
y = y[~outlier_mask.values]

cat_cols = [c for c in X_all.columns if is_string(X_all[c])]
num_feats = [c for c in X_all.columns if c not in cat_cols]
print(f"{len(num_feats)} numeric, {len(cat_cols)} categorical features")

# ----------------------------------------------------------------------------
# 4. Two views of the same data
# ----------------------------------------------------------------------------
# (a) one-hot + scaled, for linear / kernel models
all_cat = pd.concat([X_all[cat_cols], X_test[cat_cols]], axis=0)
dummies = pd.get_dummies(all_cat, dummy_na=False).astype(np.float32)
Xo_train = pd.concat(
    [X_all[num_feats].reset_index(drop=True), dummies.iloc[: len(X_all)].reset_index(drop=True)], axis=1
)
Xo_test = pd.concat(
    [X_test[num_feats].reset_index(drop=True), dummies.iloc[len(X_all):].reset_index(drop=True)], axis=1
)
Xo_train = Xo_train.replace([np.inf, -np.inf], np.nan).fillna(0)
Xo_test = Xo_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# (b) native categorical dtypes, for tree models
Xt_train = X_all.copy()
Xt_test = X_test.copy()
for c in cat_cols:
    cats = pd.Index(sorted(set(Xt_train[c].astype(str)) | set(Xt_test[c].astype(str))))
    Xt_train[c] = pd.Categorical(Xt_train[c].astype(str), categories=cats)
    Xt_test[c] = pd.Categorical(Xt_test[c].astype(str), categories=cats)
Xt_train = Xt_train.fillna(0)
Xt_test = Xt_test.fillna(0)

# ----------------------------------------------------------------------------
# 5. Models
# ----------------------------------------------------------------------------
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

cat_idx = [Xt_train.columns.get_loc(c) for c in cat_cols]

models = {
    "lasso": make_pipeline(
        RobustScaler(),
        LassoCV(alphas=np.logspace(-5, 1, 120), cv=5, max_iter=50000, random_state=SEED, n_jobs=-1),
    ),
    "enet": make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            alphas=np.logspace(-5, 1, 60), cv=5, max_iter=50000,
            random_state=SEED, n_jobs=-1,
        ),
    ),
    "ridge": make_pipeline(
        RobustScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 60)),
    ),
    "svr": make_pipeline(
        RobustScaler(),
        SVR(C=20, epsilon=0.008, gamma=0.0003),
    ),
    "lgbm": LGBMRegressor(
        n_estimators=3000, learning_rate=0.02, num_leaves=10,
        colsample_bytree=0.5, subsample=0.7, subsample_freq=1,
        min_child_samples=12, reg_lambda=1.0, random_state=SEED, verbose=-1,
    ),
    "xgb": XGBRegressor(
        n_estimators=3000, learning_rate=0.02, max_depth=3,
        subsample=0.7, colsample_bytree=0.45, min_child_weight=2,
        reg_lambda=1.0, tree_method="hist", enable_categorical=True,
        random_state=SEED, n_jobs=-1,
    ),
    "cat": CatBoostRegressor(
        iterations=3000, learning_rate=0.025, depth=4, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=SEED, verbose=0, allow_writing_files=False,
        bootstrap_type="Bernoulli", subsample=0.8,
    ),
}

TREE = {"lgbm", "xgb", "cat"}
USE_CAT = {"lgbm": "auto", "xgb": True, "cat": cat_idx}

# ----------------------------------------------------------------------------
# 6. 10-fold OOF
# ----------------------------------------------------------------------------
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
oof = pd.DataFrame(index=range(len(X_all)), columns=list(models), dtype=float)
test_pred = pd.DataFrame(0.0, index=range(len(X_test)), columns=list(models))

for name, proto in models.items():
    scores = []
    for fold, (tr_i, va_i) in enumerate(kf.split(X_all)):
        if name in TREE:
            Xtr, Xva, Xte = Xt_train.iloc[tr_i], Xt_train.iloc[va_i], Xt_test
        else:
            Xtr, Xva, Xte = Xo_train.iloc[tr_i], Xo_train.iloc[va_i], Xo_test

        m = clone(proto)
        if name in ("lgbm", "xgb", "cat"):
            if name == "cat":
                m.fit(Xtr, y[tr_i], cat_features=cat_idx,
                      eval_set=(Xva, y[va_i]), early_stopping_rounds=200)
            elif name == "lgbm":
                m.fit(Xtr, y[tr_i], eval_set=[(Xva, y[va_i])], eval_metric="rmse",
                      callbacks=[__import__("lightgbm").early_stopping(200, verbose=False)])
            else:
                m.fit(Xtr, y[tr_i], eval_set=[(Xva, y[va_i])], verbose=False)
        else:
            m.fit(Xtr, y[tr_i])

        p = m.predict(Xva)
        oof.loc[va_i, name] = p
        scores.append(rmse(y[va_i], p))
        test_pred[name] += m.predict(Xte) / kf.get_n_splits()

    print(f"{name:6s} CV RMSLE {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

single = {
    n: float(rmse(y, oof[n].values)) for n in models
}

# ----------------------------------------------------------------------------
# 7. Stack: non-negative ridge on OOF predictions
# ----------------------------------------------------------------------------
meta = Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(oof.values, y)
stack_oof = meta.predict(oof.values)
stack_score = float(rmse(y, stack_oof))
weights = dict(zip(models, np.round(meta.coef_, 3)))
print("\nstacking weights:", weights)
print(f"STACK  CV RMSLE {stack_score:.5f}")

# simple geometric mean of the three best diverse models, as a sanity blend
geo_cols = ["lasso", "svr", "cat"]
geo_oof = oof[geo_cols].mean(axis=1).values  # log space mean == geometric mean of prices
geo_score = float(rmse(y, geo_oof))

best_score = min(stack_score, geo_score)
final_test = (
    meta.predict(test_pred.values) if stack_score <= geo_score
    else test_pred[geo_cols].mean(axis=1).values
)
final_price = np.clip(np.expm1(final_test), 30000, 800000)

sub = pd.DataFrame({"Id": test_ids, "SalePrice": final_price})
sub.to_csv(OUT / "submission.csv", index=False)
print(f"\nwrote {OUT/'submission.csv'}  rows={len(sub)}")

report = {
    "n_train": int(len(X_all)),
    "n_features": int(Xo_train.shape[1]),
    "single_model_cv": {k: round(v, 5) for k, v in single.items()},
    "stack_cv": round(stack_score, 5),
    "geometric_blend_cv": round(geo_score, 5),
    "chosen": "stack" if stack_score <= geo_score else "geometric",
    "weights": {k: float(v) for k, v in weights.items()},
}
(OUT / "cv_report.json").write_text(json.dumps(report, indent=2))
np.savez(OUT / "oof_test.npz", oof=oof.values, test=test_pred.values,
         y=y, ids=test_ids.values, models=np.array(list(models)),
         ytrain_pred=stack_oof)
print(json.dumps(report, indent=2))
