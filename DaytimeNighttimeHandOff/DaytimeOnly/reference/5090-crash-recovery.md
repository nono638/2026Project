# 5090 Crash Recovery — task-055 Ollama judge run (2026-05-06)

> Last night the 5090 finished the `gemma4:31b` judge pass on `experiment_0_v3`
> and crashed partway through `qwen3.6:27b`. Work was moved out of the Dropbox
> folder due to a venv issue, so nothing synced back. Goal: locate the data
> on the 5090 and get it onto `origin/main` so the planning laptop can see it.

Run these on the **5090** in PowerShell. Stop at the first step that fails and
ping the planning instance.

---

## Step 1 — Find every project copy on the disk

There may be two: the original Dropbox-synced copy (with the broken venv) and
the moved copy where last night's run actually happened.

```powershell
Get-ChildItem -Path C:\,D:\ -Filter "run_experiment_0.py" -Recurse `
    -ErrorAction SilentlyContinue 2>$null |
  Select-Object FullName, LastWriteTime
```

Note every path that prints. The one with the most recent `LastWriteTime` on
`results/experiment_0_v3/raw_scores.csv` is the live copy.

```powershell
# Adjust the paths from above
Get-Item "<path1>\results\experiment_0_v3\raw_scores.csv",
         "<path2>\results\experiment_0_v3\raw_scores.csv" |
  Select-Object FullName, Length, LastWriteTime
```

The live copy should be dated **2026-05-06** or **2026-05-07**, not 2026-04-30.

---

## Step 2 — In the live copy, check git state

```powershell
cd <live-copy-path>
git remote -v
git status
git log --oneline origin/main..HEAD
git stash list
```

Decision tree on what you find:

- **Remote is `https://github.com/nono638/2026Project.git`** → good, continue.
  Different URL → `git remote set-url origin https://github.com/nono638/2026Project.git`.
- **`git log origin/main..HEAD` lists commits** → the run committed but never
  pushed. Skip to Step 4.
- **Working tree dirty (modified `raw_scores.csv`, untracked checkpoint)** →
  the run never committed. Continue to Step 3.

---

## Step 3 — Inspect the actual judge data before committing

Confirm the gemma4 pass landed and see how far qwen got:

```powershell
# Header — should now include gemma4 (and possibly partial qwen3.6) columns
(Get-Content results\experiment_0_v3\raw_scores.csv -TotalCount 1) -split ',' |
  Where-Object { $_ -match 'gemma4|qwen3_6|ollama' }

# Row count — 500 means full sweep, less means it stopped mid-judge
(Get-Content results\experiment_0_v3\raw_scores.csv | Measure-Object -Line).Lines

# Checkpoint header — what was the LAST judge being written when it crashed?
(Get-Content results\experiment_0_v3\raw_scores_checkpoint.csv -TotalCount 1) -split ',' |
  Where-Object { $_ -match 'gemma4|qwen3_6|ollama' }

(Get-Content results\experiment_0_v3\raw_scores_checkpoint.csv | Measure-Object -Line).Lines

# Metadata — should now list ollama judges with n_scored values
Get-Content results\experiment_0_v3\metadata.json
```

Expected: `raw_scores.csv` has full gemma4 columns and either zero or partial
qwen3.6 columns. Checkpoint has whatever was being written at crash time.

---

## Step 4 — Commit and push the recovery branch

Don't push to `main` directly — give it its own branch so the planning laptop
can review the diff first.

```powershell
git checkout -b night/task-055-v3-judge-results

git add results/experiment_0_v3/raw_scores.csv `
        results/experiment_0_v3/raw_scores_checkpoint.csv `
        results/experiment_0_v3/metadata.json `
        results/experiment_0_v3/run_v3.log

# Anything else changed? (e.g., gallery regen, tracker.json)
git status

# If tracker.json or other files changed, add them too. Then:
git commit -m "task-055: gemma4:31b complete, qwen3.6:27b partial (crashed mid-run)"
git push -u origin night/task-055-v3-judge-results
```

If push prompts for credentials and Git Credential Manager doesn't open a
browser, fall back to `gh auth login` or a Personal Access Token from
https://github.com/settings/tokens (use it as the password).

---

## Step 5 — If commit-and-push isn't viable

Fallback if git is wedged on the 5090 (auth broken, repo corrupt, etc.):

```powershell
# Tar up just the new data files and copy via Dropbox or USB
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
Compress-Archive -Path results\experiment_0_v3\raw_scores.csv, `
                       results\experiment_0_v3\raw_scores_checkpoint.csv, `
                       results\experiment_0_v3\metadata.json, `
                       results\experiment_0_v3\run_v3.log `
                 -DestinationPath "$HOME\Dropbox\v3-recovery-$stamp.zip"
```

The planning laptop will pick up the zip via Dropbox and unpack it manually.

---

## Step 6 — Back on the planning laptop

Once Step 4 lands on GitHub:

```powershell
git fetch origin
git log --oneline origin/night/task-055-v3-judge-results
git checkout night/task-055-v3-judge-results -- results/experiment_0_v3/
git diff --stat main -- results/experiment_0_v3/
```

Then re-run the gallery and confirm the new judge rows in
`results/experiment_0_v3/report.md`:

```powershell
python scripts\generate_gallery.py
```

If the gemma4 pass is clean, decide whether to (a) merge as-is and re-launch
qwen3.6 fresh on the 5090, or (b) resume qwen3.6 from the checkpoint.
`run_experiment_0.py` should handle resume via existing CSV columns —
verify by re-reading task-055's spec before relaunching.

---

## Notes for the planning instance

- The 5090's `.venv/` lives outside Dropbox now. Don't blow it away —
  it's the working one. Update the runbook (Step 2) so future setups
  recreate the venv at a non-Dropbox path from the start.
- If the 5090 had no commits ahead of origin AND the working tree was
  clean, the run wrote to a different path entirely (or never wrote at
  all). Search for `raw_scores_checkpoint.csv` outside the project root:
  `Get-ChildItem -Path C:\,D:\ -Filter "raw_scores_checkpoint.csv" -Recurse -ErrorAction SilentlyContinue`.
