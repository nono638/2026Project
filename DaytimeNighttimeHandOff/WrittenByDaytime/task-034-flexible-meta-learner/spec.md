# task-034: Flexible Meta-Learner — User-Defined Objectives

## Summary

Redesign `src/model/train.py` and `src/model/predict.py` so users can define their own
training objective instead of being locked into hardcoded multiclass classification.
The new API supports regression (predict a score), classification (predict best config),
and constrained optimization (best config within constraints) — all through a single
`train()` function that auto-detects mode from the target column.

This is the foundation for RAGBench's "best is user-defined" philosophy: users choose
what to optimize (quality, latency, cost, smallest model) and the meta-learner trains
to that objective. XGBoost handles both regression and classification natively.

## Requirements

1. `train()` accepts a `target` parameter (column name to optimize, default `"quality"`)
2. `train()` accepts a `mode` parameter: `"regression"`, `"classification"`, or `None` (auto-detect)
3. `train()` accepts an `objective` parameter: `"maximize"` or `"minimize"` (for classification mode)
4. `train()` accepts a `features` parameter: list of X column names (defaults to `FEATURE_COLS`)
5. `train()` accepts a `constraints` parameter: dict of column filters like `{"quality": ">3.0", "model_param_billions": "<4"}`
6. **Auto-detection**: if `target="config"` → classification (backward compat). If target is a numeric column and no explicit mode → regression. If target is numeric + `mode="classification"` → "which config gives best {target}?"
7. **Classification flow**: apply constraints → group by `(query_text, doc_title)` → per group, pick winner via `objective(target)` → winner's config label is Y → train XGBClassifier
8. **Regression flow**: apply constraints → encode config columns as categorical X features alongside FEATURE_COLS → every row is a training example → target column is Y → train XGBRegressor
9. **Saved artifacts** include a `meta.json` file recording mode, target, objective, constraints, and feature columns — so `predict()` knows how to behave at inference time
10. `predict()` auto-adapts based on saved `meta.json`: classification returns `{chunker, embedder, strategy, model, confidence}`, regression returns `{predicted_value, model_type: "regression"}`
11. **Backward compatibility**: calling `train(df)` with no extra args works like before (classification, target=quality, maximize, threshold mapped to constraints). The old `quality_threshold` param is kept but internally mapped to `constraints={"quality": f">={quality_threshold}"}`
12. `src/config.py` retains `FEATURE_COLS` and `DEFAULT_QUALITY_THRESHOLD` unchanged — they're defaults, not hardcoded behaviors
13. All existing tests in `tests/test_model.py` continue to pass (backward compat)

## Files to Modify

### `src/model/train.py` — Major rewrite
- **`train()` function**: new signature with `target`, `mode`, `objective`, `features`, `constraints` params. Dispatch to `_train_classifier()` or `_train_regressor()` internally.
- **`prepare_data()`**: rename to `_prepare_classification_data()`. Add `target`, `objective`, `constraints` params. The grouping + winner-selection logic stays here but becomes configurable.
- **New `_prepare_regression_data()`**: encodes config columns (chunker, embedder, strategy, model) as pandas categoricals, combines with feature columns, returns X and y.
- **New `_apply_constraints()`**: parses constraint strings (`">3.0"`, `"<4"`, `">=2.5"`, `"==NaiveRAG"`) and filters the DataFrame. Supports `>`, `>=`, `<`, `<=`, `==`, `!=` operators. Numeric values parsed as float; string values compared as strings.
- **New `_save_meta()`**: writes `meta.json` alongside model artifacts recording all training parameters.
- **Keep** the `FEATURE_COLS` re-export for backward compat (`from src.model.train import FEATURE_COLS` is used in tests).

### `src/model/predict.py` — Adapt to mode
- **`_load_model()`**: also loads `meta.json` to determine mode. If `meta.json` is missing (old model), assume classification mode for backward compat.
- **`predict()`**: branch on mode. Classification path unchanged. Regression path: build feature row including encoded config columns, return `{predicted_value, model_type}`.
- **New `predict_all_configs()`** (classification only): given features, return all configs ranked by confidence (not just top-1). Useful for constrained optimization at inference time.

### `src/app.py` — Minor update
- The `/recommend` endpoint currently calls `predict(features)` and expects classification output. No change needed — the default training mode is classification, so default models produce classification output. Add a note in the docstring that the endpoint assumes a classification model is loaded.

### `tests/test_model.py` — Extend with new test classes
- Keep all existing test classes unchanged (backward compat verification)
- Add `TestRegressionMode` class
- Add `TestClassificationWithConstraints` class
- Add `TestAutoDetection` class
- Add `TestPredictRegression` class

## New Dependencies

None — XGBoost, sklearn, and pandas already support everything needed. `XGBRegressor` is in the same `xgboost` package as `XGBClassifier`.

## Edge Cases

1. **Constraints filter out ALL rows**: raise `ValueError("No rows remain after applying constraints: {constraints}")` with the constraints dict in the message.
2. **Constraints filter out all rows in a group** (classification): skip that group. If ALL groups are empty after constraints, raise ValueError.
3. **Only 1 unique target class after constraints** (classification): sklearn can't stratify. Fall back to non-stratified split with a warning printed to stdout.
4. **Target column doesn't exist**: raise `ValueError(f"Target column '{target}' not found in data. Available columns: {list(df.columns)}")`.
5. **Invalid constraint operator**: raise `ValueError(f"Invalid constraint format: '{constraint}'. Expected operator + value, e.g., '>3.0', '<=5', '==NaiveRAG'")`.
6. **Regression with no config columns in data**: if chunker/embedder/strategy/model columns are missing, train on FEATURE_COLS only (features param controls this).
7. **`meta.json` missing at predict time** (old model): assume classification mode. Print a warning: `"No meta.json found — assuming classification mode (pre-task-034 model)"`.
8. **Empty features list**: raise `ValueError("features list cannot be empty")`.
9. **`target="config"` explicit**: treat as classification with the old 4-axis config label behavior. `objective` defaults to `"maximize"` on `"quality"` column for winner selection.

## Decisions Made

- **Single `train()` entry point, not separate functions**: one function that dispatches internally. **Why:** users learn one API. Internal helpers (`_train_classifier`, `_train_regressor`) keep the code clean.
- **Constraints as string expressions, not callables**: `{"quality": ">3.0"}` not `{"quality": lambda x: x > 3.0}`. **Why:** strings are JSON-serializable (saved in meta.json for reproducibility), and the supported operators cover all practical cases.
- **Config columns encoded as pandas categoricals for regression**: use `pd.Categorical` + `.cat.codes` for chunker/embedder/strategy/model when they're X features. **Why:** XGBoost handles integer-encoded categoricals natively. One-hot encoding would create too many columns with the 4-axis combinatorial space.
- **`meta.json` for mode persistence**: saved alongside model artifacts. **Why:** predict() needs to know what mode the model was trained in without the user passing it again. Also enables reproducibility — you can see exactly how a model was trained.
- **Backward compat via defaults**: `train(df)` works exactly as before because defaults are `target="quality"`, `mode=None` (auto-detects classification since the old `quality_threshold` logic applies), `objective="maximize"`. **Why:** don't break existing code or tests.
- **`predict_all_configs()` as separate function**: not merged into `predict()`. **Why:** different return type (list vs dict). Keeps `predict()` simple for the common case.
- **`_apply_constraints` supports string equality**: `"==NaiveRAG"` for filtering by strategy name. **Why:** users need to constrain by categorical values too, not just numerics.

## What NOT to Touch

- **`src/features.py`** — feature extraction is independent of training mode
- **`src/config.py`** — `FEATURE_COLS` and `DEFAULT_QUALITY_THRESHOLD` stay as-is (they're defaults)
- **`src/experiment.py`** — experiment output format doesn't change
- **`src/metadata.py`** — parsing functions are unchanged
- **Existing tests in `tests/test_model.py`** — the existing `TestPrepareData`, `TestTrain`, `TestPredict` classes must pass unchanged (backward compat). Add new test classes alongside them.

## Testing Approach

Pre-written tests in `DaytimeNighttimeHandOff/WrittenByDaytime/task-034-flexible-meta-learner/tests/`.
Copy to `tests/test_model.py` (append new classes to existing file).

Tests cover:
- Regression mode: train + predict round-trip, correct artifact files, numeric predictions
- Classification with constraints: filters applied, winner selection respects objective
- Classification with minimize objective (e.g., smallest model)
- Auto-detection: numeric target → regression, "config" → classification, numeric + explicit mode → classification
- Constraint edge cases: all rows filtered, invalid operator, string equality
- Backward compat: existing `train(df)` call still works
- `predict_all_configs()`: returns ranked list
- `meta.json`: saved and loaded correctly, missing meta.json falls back to classification

Run with: `pytest tests/test_model.py -v`
