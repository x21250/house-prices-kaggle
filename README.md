# House Prices — Advanced Regression Techniques (Kaggle)

Feature engineering + 7-model out-of-fold stack for the Ames housing dataset.

| | |
|---|---|
| Public LB | **0.12164** |
| Best OOF (10-fold RMSLE) | 0.10500 |
| Best single model OOF | 0.10984 (Lasso) |

The OOF/CV number is **optimistic by ~0.016** relative to the leaderboard. Treat it as a
model-selection signal, not as a prediction of your score. See "What the gap means" below.

## Files

| file | role |
|---|---|
| `train_pipeline.py` | main pipeline: prep → 10-fold OOF → non-negative ridge stack → `submission.csv` |
| `train_pipeline_v2.py` | experimental: repeated CV, ExtraTrees, Huber GB (not validated; see below) |
| `requirements.txt` | pinned environment |
| `cv_report.json` | per-model OOF scores and stack weights |

## Pipeline

1. **NA semantics.** In this dataset a missing `PoolQC` means *no pool*, not *unknown*, so
   the 15 such columns are filled with `"None"`. `LotFrontage` is imputed by
   `Neighborhood` median; `GarageYrBlt`/`MasVnrArea`/basement areas are filled with 0.
2. **Ordinal encoding** of the 21 quality columns (`Po < Fa < TA < Gd < Ex`). One-hot
   would discard the ordering, which is most of the signal.
3. **Nominal-as-text.** `MSSubClass` (20/30/…/190) and `MoSold` are codes, not quantities.
4. **Engineered:** `TotalSF`, `TotalBath`, `TotalPorch`, `Age`, `RemodAge`, `QualSF`,
   `OverallGrade`, `LivLotRatio`, plus has-pool / has-basement / has-garage flags.
5. **Target:** `log1p(SalePrice)`; metric is RMSE in log space, so this is the right scale.
6. **Outliers:** drop the two `GrLivArea > 4000` cheap sales (the classic Ames points).
7. **Models:** LassoCV, ElasticNetCV, RidgeCV, SVR(RBF), LightGBM, XGBoost, CatBoost.
   Tree models get native categorical dtypes; linear/kernel models get one-hot.
8. **Stack:** `Ridge(alpha=1, positive=True)` on the 7 OOF columns, fit in log space.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download train.csv / test.csv from the competition's Data tab into ./data/
python train_pipeline.py
```

Edit `DATA` at the top of `train_pipeline.py` to point at the data directory.

## What the gap means

OOF 0.105 vs LB 0.1216 is not a blending problem: the stack, a 3-model geometric blend and
Lasso alone differ by only ~0.008 in log-price from each other, roughly half the gap. The
candidates are:

- **CV optimism.** 1458 rows, 10 folds, and `LassoCV` choosing its alpha inside each fold.
- **Booster overfitting.** Early stopping peeks at the validation fold on a very small train set.

`submission-lasso.csv` (Lasso only) is the experiment that separates the two.

Legitimate solutions in this competition cluster around 0.115–0.118 public. The 0.00000
entries at the top of the leaderboard are not models — people matched rows back to Dean
De Cock's original Ames dataset and copied real sale prices.

## Environment note (macOS)

`lightgbm` and `xgboost` wheels link against `libomp.dylib` at
`/opt/homebrew/opt/libomp/lib/`, which does not exist without Homebrew. There is no pip
package for it. The workaround used here is to point both dylibs at the copy bundled with
scikit-learn's wheel:

```bash
OMP="$(python -c 'import sklearn,os;print(os.path.dirname(sklearn.__file__))')/.dylibs/libomp.dylib"
install_name_tool -change @rpath/libomp.dylib "$OMP" \
  "$(python -c 'import lightgbm,os;print(os.path.dirname(lightgbm.__file__))')/lib/lib_lightgbm.dylib"
install_name_tool -change @rpath/libomp.dylib "$OMP" \
  "$(python -c 'import xgboost,os;print(os.path.dirname(xgboost.__file__))')/lib/libxgboost.dylib"
```

This must be re-applied after reinstalling either package.

## Status of `train_pipeline_v2.py`

It was killed by the OS (exit 137) during its first run — repeated CV × 8 models with
`ExtraTrees(n_jobs=-1)` exceeded the available memory. **No v2 numbers exist.** It has not
been validated; do not cite it.
