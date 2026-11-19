# House Prices — Advanced Regression Techniques (Kaggle)

Two pipelines for the same competition, kept side by side so the difference in score is
traceable to specific decisions rather than to mystery.

| | public LB | CV (RMSLE) |
|---|---|---|
| Ensemble pipeline (`src/train_pipeline.py`) | **0.12164** | 0.10500 |
| Cleaned notebook pipeline (`src/house_prices_clean.py`) | not submitted | 0.12597 |
| Lasso baseline, same preprocessing | — | 0.14385 |

CV is optimistic by roughly 0.016 in this competition. Treat it as a signal for choosing
between models, not as a forecast of your leaderboard position.

## Layout

```
src/train_pipeline.py         feature engineering + 7-model OOF stack -> submission.csv
src/house_prices_clean.py     the original notebook, repaired, as a runnable script
src/train_pipeline_v2.py      experiment that was killed by the OS; never validated
notebooks/house_prices.ipynb  sectioned, commented version of house_prices_clean.py
reports/cv_report.json        per-model OOF scores and stack weights
```

## The two pipelines

**`train_pipeline.py`** is the ensemble. Log target, ordinal encoding of the 21 quality
columns, `NA → "None"` semantics, `LotFrontage` by neighbourhood median, engineered
`TotalSF` / `TotalBath` / `Age` / `QualSF` features, two outlier sales dropped, then
Lasso / ElasticNet / Ridge / SVR / LightGBM / XGBoost / CatBoost stacked with a
non-negative ridge on out-of-fold predictions.

**`house_prices_clean.py`** is the original notebook approach: `Pipeline` +
`ColumnTransformer` + `GridSearchCV` over XGBoost, with one-hot encoding and a `LassoCV`
baseline. Same model family, correct code. It scores 0.126, which is where the 0.14548
leaderboard entry came from once optimism is accounted for.

The gap between 0.126 and 0.105 is feature engineering and stacking, not a bigger grid.

## Bugs fixed when cleaning the notebook

1. **Log target used inconsistently.** The grid was fit on `np.log(y)`, but `predict()`
   (still log-scale) went straight into the CSV, and the scorer was
   `neg_root_mean_squared_log_error` — which takes `log1p` of an already-logged value.
   Now: fit on `log1p`, score with plain RMSE (identical on that scale), `expm1` on output.
2. **`OrdinalEncoder` on every categorical.** That assigns an arbitrary order to
   `Neighborhood` and hands the linear baseline numbers where it needs indicators.
   Replaced with one-hot.
3. **Step named `rf` holding an XGBoost.** Every param key read `rf__learning_rate`.
   Renamed to `model`.
4. **432 grid candidates × 5 folds = 2160 fits**, most of them near-duplicates.
   Trimmed to 18 × 5.
5. **Dead cell** re-assigned `pipeline` to a `LassoCV` that was never used; `best_model`
   still pointed at the grid search.
6. **Submission columns** were `id` / `test['Id']`; Kaggle expects `Id`.
7. **`select_dtypes(include=['object'])` returns nothing under pandas 3**, where strings
   are dtype `str`. Column detection now uses `dtype.kind`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/house_prices_clean.py     # ~90 s, writes submission_notebook.csv
python src/train_pipeline.py         # ~2 min, writes submission.csv
```

Both expect the competition CSVs in `~/Downloads/house-prices-advanced-regression-techniques`;
edit `DATA_DIR` at the top of either file to change that.

## Environment note (macOS)

`lightgbm` and `xgboost` wheels link against `libomp.dylib` at
`/opt/homebrew/opt/libomp/lib/`, which does not exist without Homebrew, and there is no
pip package for it. Workaround — point both at the copy bundled inside scikit-learn:

```bash
OMP="$(python -c 'import sklearn,os;print(os.path.dirname(sklearn.__file__))')/.dylibs/libomp.dylib"
install_name_tool -change @rpath/libomp.dylib "$OMP" \
  "$(python -c 'import lightgbm,os;print(os.path.dirname(lightgbm.__file__))')/lib/lib_lightgbm.dylib"
install_name_tool -change @rpath/libomp.dylib "$OMP" \
  "$(python -c 'import xgboost,os;print(os.path.dirname(xgboost.__file__))')/lib/libxgboost.dylib"
```

Re-apply after reinstalling either package.

## Note on the leaderboard

Legitimate entries cluster around 0.115–0.118 public. The 0.00000 scores at the top are not
models: people matched rows back to Dean De Cock's original Ames dataset and copied the
real sale prices.
