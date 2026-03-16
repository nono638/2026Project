# Task 004: ExperimentResult Analysis and Visualization

**Depends on:** task-002 (needs working ExperimentResult class)

## What
Flesh out the `ExperimentResult` class with proper analysis, comparison, and
visualization methods. This is what makes the tool useful for research — users
need to be able to quickly understand their results.

## Why
The current `ExperimentResult` has skeleton `compare()` and `pivot()` methods.
A research tool needs richer analysis: statistical comparisons, heatmaps,
per-query breakdowns, and export formats.

## Exact Changes

### Modify `src/experiment.py` — expand `ExperimentResult`

Add these methods to the existing `ExperimentResult` class:

#### `summary() -> pd.DataFrame`
Group by all four axes (chunker, embedder, strategy, model) and compute mean, std,
min, max, count for all score columns (faithfulness, relevance, conciseness, quality).
Print and return the result.

#### `compare_strategies(metric: str = "quality") -> pd.DataFrame`
Group by strategy only, aggregate metric. Print ranked table. This is the quick
"which strategy wins" view.

#### `compare_models(metric: str = "quality") -> pd.DataFrame`
Group by model only, aggregate metric. Print ranked table.

#### `heatmap(rows: str, cols: str, values: str = "quality", save_path: Path | None = None)`
Create a matplotlib heatmap of the pivot table. Use `plt.imshow()` or `sns.heatmap()`
(prefer matplotlib to avoid adding seaborn dependency).

Implementation:
```python
def heatmap(self, rows: str, cols: str, values: str = "quality",
            save_path: Path | None = None):
    import matplotlib.pyplot as plt

    pivot = self.pivot(rows, cols, values)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells with values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center")

    plt.colorbar(im)
    plt.title(f"{values} by {rows} × {cols}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
```

#### `per_query(metric: str = "quality") -> pd.DataFrame`
For each unique query, show the best and worst config and the score spread.
Useful for identifying queries where strategy choice matters most.

```python
def per_query(self, metric: str = "quality") -> pd.DataFrame:
    results = []
    for query, group in self.df.groupby("query_text"):
        best = group.loc[group[metric].idxmax()]
        worst = group.loc[group[metric].idxmin()]
        results.append({
            "query": query[:80],
            "best_config": f"{best['strategy']}/{best['model']}",
            "best_score": best[metric],
            "worst_config": f"{worst['strategy']}/{worst['model']}",
            "worst_score": worst[metric],
            "spread": best[metric] - worst[metric],
        })
    result_df = pd.DataFrame(results).sort_values("spread", ascending=False)
    print(result_df.to_string(index=False))
    return result_df
```

#### `strategy_vs_size(metric: str = "quality") -> pd.DataFrame`
The key analysis for our research question: pivot strategy × model, showing where
small models + smart strategies beat large models + simple strategies.

```python
def strategy_vs_size(self, metric: str = "quality") -> pd.DataFrame:
    """The core research question: when does strategy beat size?"""
    pivot = self.pivot("strategy", "model", metric)
    print(pivot.round(3).to_string())
    return pivot
```

#### `to_csv(path: Path) -> None`
Export to CSV for sharing or further analysis in R/Excel.

#### `merge(other: "ExperimentResult") -> "ExperimentResult"`
Combine two ExperimentResults (from different runs). Just concatenate DataFrames.
```python
def merge(self, other: "ExperimentResult") -> "ExperimentResult":
    return ExperimentResult(pd.concat([self.df, other.df], ignore_index=True))
```

### Add `import matplotlib` to requirements

matplotlib is already in the system Python but may need to be in the venv.
Add a check at the top of the heatmap method:
```python
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib required for plotting: pip install matplotlib")
```

## Tests to Add: `tests/test_analysis.py`

Create a fixture that builds a synthetic ExperimentResult DataFrame with known values
(don't run actual experiments — just construct the DataFrame directly):

```python
@pytest.fixture
def sample_results():
    rows = []
    for strategy in ["naive", "self_rag"]:
        for model in ["qwen3:0.6b", "qwen3:4b"]:
            for i in range(5):
                rows.append({
                    "doc_title": f"doc_{i}",
                    "query_text": f"query_{i}",
                    "query_type": "lookup",
                    "chunker": "semantic:mxbai-embed-large",
                    "embedder": "ollama:mxbai-embed-large",
                    "model": model,
                    "strategy": strategy,
                    "answer": "test answer",
                    "faithfulness": 3.0 + (0.5 if strategy == "self_rag" else 0),
                    "relevance": 3.5,
                    "conciseness": 4.0,
                    "quality": 3.5 + (0.5 if strategy == "self_rag" else 0),
                    "query_length": 10,
                    "num_named_entities": 1,
                    "doc_length": 500,
                    "doc_vocab_entropy": 8.5,
                    "mean_retrieval_score": 0.7,
                    "var_retrieval_score": 0.02,
                    "timestamp": "2026-03-16T12:00:00",
                })
    return ExperimentResult(pd.DataFrame(rows))
```

Test cases:
1. `test_compare` — returns DataFrame with expected groups
2. `test_compare_strategies` — strategy ranking correct (self_rag > naive in fixture)
3. `test_pivot` — correct shape and values
4. `test_per_query` — correct spread calculation
5. `test_strategy_vs_size` — pivot has right shape
6. `test_parquet_roundtrip` — save and load preserves data
7. `test_csv_export` — file is created and readable
8. `test_merge` — two results combined correctly
9. `test_heatmap_saves_to_file` — use `save_path` to a tmp file, verify file exists (don't check image content)

## What NOT to Touch
- `src/protocols.py`, `src/retriever.py` — core framework
- `src/chunkers/`, `src/embedders/`, `src/strategies/`, `src/scorers/` — components
- `src/model/`, `src/app.py`
