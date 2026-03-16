# Plan: task-004 — ExperimentResult Analysis and Visualization

## Files to Modify
1. `src/experiment.py` — add analysis methods to ExperimentResult class

## Files to Create
1. `tests/test_analysis.py` — tests with synthetic data fixture

## Methods to Add to ExperimentResult
- summary()
- compare_strategies(metric)
- compare_models(metric)
- heatmap(rows, cols, values, save_path)
- per_query(metric)
- strategy_vs_size(metric)
- to_csv(path)
- merge(other)

## Approach
1. Read current experiment.py
2. Add all new methods to ExperimentResult
3. Install matplotlib if needed
4. Write tests
5. Run all tests

## Ambiguities
- None — spec is very detailed with exact implementations.
