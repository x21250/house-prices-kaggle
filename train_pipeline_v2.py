"""
v2: same skeleton as v1, with the feature-semantics fixes and more model diversity.

Changes vs v1
  - MSSubClass treated as the nominal code it is (20/30/.../190), not a number
  - MoSold as categorical + cyclical sin/cos
  - GarageAge, TotalFinSF, deck/porch flags, OverallQual^2
  - 2 x 10-fold repeated OOF (stabler meta weights than a single 10-fold)
  - Ridge dropped (v1 gave it weight 0.0); ExtraTrees + Huber GB added for diversity
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

DATA = Path.home() / "Downloads/house-prices-advanced-regression-techniques"
OUT = Path(__file__).resolve().parent
SEED = 42


def is_string(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype.name in ("object", "category")


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
test_ids = test["Id"].copy()
y = np.log1p(train["SalePrice"].values)
train = train.drop(columns=["SalePrice"])
n_train = len(train)
df = pd.concat([train, test], axis=0, ignore_index=True)

# --- NA semantics -----------------------------------------------------------
for c in ["GarageYrBlt", "GarageCars", "GarageArea", "MasVnrArea",
          "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
          "BsmtFullBath", "BsmtHalfBath"]:
    df[c] = df[c].fillna(0)

df.loc[df["GarageYrBlt"] > df["YrSold"], "GarageYrBlt"] = df["YrSold"]

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda s: s.fillna(s.median()))
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

for c in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
          "GarageType", "GarageFinish", "GarageQual", "GarageCond",
          "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
          "MasVnrType"]:
    df[c] = df[c].fillna("None")

for c in df.columns:
    if is_string(df[c]) and df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

# --- ordinal ----------------------------------------------------------------
q5 = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ord_maps = {c: q5 for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                            "HeatingQC", "KitchenQual", "FireplaceQu",
                            "GarageQual", "GarageCond"]}
ord_maps.update({
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
})
for c, m in ord_maps.items():
    df[c] = df[c].map(m).astype(float)

# --- nominal-as-text: these are codes, not quantities ------------------------
df["MSSubClass"] = df["MSSubClass"].astype(str)
df["MoSold"] = df["MoSold"].astype(str)
df["MoSoldNum"] = pd.to_numeric(df["MoSold"])
df["MoSoldSin"] = np.sin(2 * np.pi * df["MoSoldNum"] / 12)
df["MoSoldCos"] = np.cos(2 * np.pi * df["MoSoldNum"] / 12)

# --- engineered features ----------------------------------------------------
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalFinSF"] = df["GrLivArea"] + df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                   + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
df["TotalPorch"] = (df["OpenPorchSF"] + df["3SsnPorch"] + df["EnclosedPorch"]
                    + df["ScreenPorch"] + df["WoodDeckSF"])
df["Age"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["HasDeck"] = (df["WoodDeckSF"] > 0).astype(int)
df["HasPorch"] = (df["TotalPorch"] > 0).astype(int)
df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
df["OverallQual2"] = df["OverallQual"] ** 2
df["QualSF"] = df["OverallQual"] * df["TotalSF"]
df["ExterScore"] = df["ExterQual"] * df["ExterCond"]
df["GarageScore"] = df["GarageCars"] * df["GarageArea"]
df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"].replace(0, np.nan)
df["BsmtFinRatio"] = df["BsmtFinSF1"] / df["TotalBsmtSF"].replace(0, np.nan)
df["PorchRatio"] = df["TotalPorch"] / df["GrLivArea"]
df["AvgRoomSF"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, np.nan)
df["BsmtScore"] = df["BsmtQual"].fillna(0) * df["TotalBsmtSF"]
df["KitchenScore"] = df["KitchenQual"] * df["KitchenAbvGr"]
df["AgeQual"] = df["Age"] * df["OverallQual"]

# --- skew -------------------------------------------------------------------
num_cols = [c for c in df.columns if df[c].dtype.name in ("int64", "float64", "int32", "float32")]
for c in [c for c in num_cols if abs(df[c].skew()) > 0.75]:
    if df[c].min() >= 0:
        df[c] = np.log1p(df[c])

df = df.loc[:, df.nunique(dropna=False) > 1]

X_all = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy().reset_index(drop=True)

mask = (X_all["GrLivArea"] > np.log1p(4000)) & (y < np.log(300000))
print(f"dropping {int(mask.sum())} outliers")
X_all = X_all.loc[~mask].reset_index(drop=True)
y = y[~mask.values]

cat_cols = [c for c in X_all.columns if is_string(X_all[c])]
num_feats = [c for c in X_all.columns if c not in cat_cols]
print(f"{len(num_feats)} numeric, {len(cat_cols)} categorical")

# --- views ------------------------------------------------------------------
all_cat = pd.concat([X_all[cat_cols], X_test[cat_cols]], axis=0)
dummies = pd.get_dummies(all_cat, dummy_na=False).astype(np.float32)
Xo_train = pd.concat([X_all[num_feats].reset_index(drop=True),
                      dummies.iloc[: len(X_all)].reset_index(drop=True)], axis=1)
Xo_test = pd.concat([X_test[num_feats].reset_index(drop=True),
                     dummies.iloc[len(X_all):].reset_index(drop=True)], axis=1)
Xo_train = Xo_train.replace([np.inf, -np.inf], np.nan).fillna(0)
Xo_test = Xo_test.replace([np.inf, -np.inf], np.nan).fillna(0)

Xt_train, Xt_test = X_all.copy(), X_test.copy()
for c in cat_cols:
    cats = pd.Index(sorted(set(Xt_train[c].astype(str)) | set(Xt_test[c].astype(str))))
    Xt_train[c] = pd.Categorical(Xt_train[c].astype(str), categories=cats)
    Xt_test[c] = pd.Categorical(Xt_test[c].astype(str), categories=cats)
Xt_train, Xt_test = Xt_train.fillna(0), Xt_test.fillna(0)

# --- models -----------------------------------------------------------------
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

cat_idx = [Xt_train.columns.get_loc(c) for c in cat_cols]

models = {
    "lasso": make_pipeline(
        RobustScaler(),
        LassoCV(alphas=np.logspace(-5, 1, 120), cv=5, max_iter=50000, random_state=SEED, n_jobs=-1)),
    "enet": make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     alphas=np.logspace(-5, 1, 60), cv=5, max_iter=50000,
                     random_state=SEED, n_jobs=-1)),
    "svr": make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003)),
    "xgb": XGBRegressor(n_estimators=4000, learning_rate=0.015, max_depth=3,
                        subsample=0.7, colsample_bytree=0.45, min_child_weight=2,
                        reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                        random_state=SEED, n_jobs=-1),
    "lgbm": LGBMRegressor(n_estimators=4000, learning_rate=0.015, num_leaves=10,
                          colsample_bytree=0.5, subsample=0.7, subsample_freq=1,
                          min_child_samples=12, reg_lambda=1.0,
                          random_state=SEED, verbose=-1),
    "cat": CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=4, l2_leaf_reg=3.0,
                             loss_function="RMSE", random_seed=SEED, verbose=0,
                             allow_writing_files=False, bootstrap_type="Bernoulli", subsample=0.8),
    "et": ExtraTreesRegressor(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                              random_state=SEED, n_jobs=-1),
    "gbhuber": GradientBoostingRegressor(loss="huber", n_estimators=800, learning_rate=0.03,
                                         max_depth=4, subsample=0.8, random_state=SEED),
}

# sklearn's ET / GBR can't consume pandas category dtype -> they use the one-hot view
TREE = {"xgb", "lgbm", "cat"}

N_SPLITS, N_REPEATS = 10, 2
oof_sum = pd.DataFrame(0.0, index=range(len(X_all)), columns=list(models))
test_pred = pd.DataFrame(0.0, index=range(len(X_test)), columns=list(models))
n_fits = N_SPLITS * N_REPEATS
fold_scores = {k: [] for k in models}

for rep in range(N_REPEATS):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
    for tr_i, va_i in kf.split(X_all):
        for name, proto in models.items():
            if name in TREE:
                Xtr, Xva, Xte = Xt_train.iloc[tr_i], Xt_train.iloc[va_i], Xt_test
            else:
                Xtr, Xva, Xte = Xo_train.iloc[tr_i], Xo_train.iloc[va_i], Xo_test

            m = clone(proto)
            if name == "cat":
                m.fit(Xtr, y[tr_i], cat_features=cat_idx,
                      eval_set=(Xva, y[va_i]), early_stopping_rounds=200)
            elif name == "lgbm":
                m.fit(Xtr, y[tr_i], eval_set=[(Xva, y[va_i])], eval_metric="rmse",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
            elif name == "xgb":
                m.fit(Xtr, y[tr_i], eval_set=[(Xva, y[va_i])], verbose=False)
            else:
                m.fit(Xtr, y[tr_i])

            p = m.predict(Xva)
            oof_sum.loc[va_i, name] += p
            fold_scores[name].append(rmse(y[va_i], p))
            test_pred[name] += m.predict(Xte) / n_fits
    print(f"repeat {rep + 1}/{N_REPEATS} done")

oof = oof_sum / N_REPEATS

single = {n: float(rmse(y, oof[n].values)) for n in models}
for n, s in sorted(single.items(), key=lambda kv: kv[1]):
    print(f"{n:8s} OOF {s:.5f}")

meta = Ridge(alpha=1.0, positive=True).fit(oof.values, y)
stack_oof = meta.predict(oof.values)
stack_score = float(rmse(y, stack_oof))
print("\nweights:", {k: round(float(v), 3) for k, v in zip(models, meta.coef_)})
print(f"STACK OOF {stack_score:.5f}")

final_test = meta.predict(test_pred.values)
final_price = np.clip(np.expm1(final_test), 30000, 800000)
pd.DataFrame({"Id": test_ids, "SalePrice": final_price}).to_csv(OUT / "submission_v2.csv", index=False)

report = {
    "n_features": int(Xo_train.shape[1]),
    "single_model_oof": {k: round(v, 5) for k, v in single.items()},
    "stack_oof": round(stack_score, 5),
    "weights": {k: round(float(v), 3) for k, v in zip(models, meta.coef_)},
}
(OUT / "cv_report_v2.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote submission_v2.csv  CV={stack_score:.5f}")
