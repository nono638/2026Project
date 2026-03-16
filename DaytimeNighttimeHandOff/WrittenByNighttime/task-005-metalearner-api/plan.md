# Plan: task-005 — Wire Meta-Learner and FastAPI to New Architecture

## Files to Modify
- `src/model/train.py` — Update to accept ExperimentResult/DataFrame/Parquet, 4-axis config label
- `src/model/predict.py` — Split config into 4 parts (chunker, embedder, strategy, model)
- `src/app.py` — Update imports to new architecture, add chunker/embedder to response

## Files to Create
- `tests/test_model.py` — 4 tests: prepare_data, train_runs, predict_returns_4_axes, config_label_format

## Approach
1. Merge dependency branches (task-002, task-004) into task-005 branch
2. Rewrite train.py per spec — remove src.config import, accept multiple data input types
3. Rewrite predict.py per spec — 4-part config split
4. Rewrite app.py per spec — new imports, 4-axis response, lazy default components
5. Write tests/test_model.py with synthetic DataFrame
6. Run all tests

## Ambiguities
- The spec references `src.config` constants (QUALITY_THRESHOLD, MODELS_DIR) which no longer exist
  after migration. Will use inline defaults as shown in the spec code.
- No pre-written test files in WrittenByDaytime/task-005/tests/ — will create tests/test_model.py
  per spec's test descriptions.
- The spec's `extract_features` signature takes (query, document, retriever) matching the
  task-001 features.py, not the old (query, document, chunks, index) signature.
