# Plan: task-034 — Flexible Meta-Learner

## Approach

### `src/model/train.py` — Major rewrite
1. Add `_apply_constraints()` to parse and apply constraint dicts
2. Rename existing `prepare_data()` to `_prepare_classification_data()` with new params (target, objective, constraints)
3. Add `_prepare_regression_data()` for regression mode
4. Add `_save_meta()` for meta.json persistence
5. Rewrite `train()` with new signature: target, mode, objective, features, constraints
6. Keep backward compat: `train(df)` still works, `quality_threshold` mapped to constraints
7. Keep `FEATURE_COLS` re-export

### `src/model/predict.py` — Adapt to mode
1. `_load_model()` also loads meta.json; missing meta.json = classification assumed
2. `predict()` branches on mode (classification vs regression)
3. Add `predict_all_configs()` for ranked config list

### Tests
- Append new test classes from pre-written test file to `tests/test_model.py`
- Existing tests must still pass

## Ambiguities
- `prepare_data` is imported by existing tests — keeping backward compat via the old signature as default path
- The `target="config"` auto-detection needs special handling since "config" column may not exist yet
