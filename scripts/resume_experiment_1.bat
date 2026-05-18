@echo off
cd /d "D:\Projects\2026Project"

REM ============================================================
REM  5090-STABILITY SETUP — READ BEFORE RUNNING
REM ============================================================
REM  Ollama must be running with these env vars BEFORE this bat
REM  launches. The default Ollama tray-icon service does NOT pick
REM  up env vars set here. Steps:
REM
REM  1. Right-click the Ollama tray icon → Quit
REM  2. Open a NEW terminal and run:
REM       set OLLAMA_KEEP_ALIVE=30m
REM       set OLLAMA_MAX_LOADED_MODELS=2
REM       set OLLAMA_NUM_PARALLEL=1
REM       set OLLAMA_MAX_QUEUE=1
REM       ollama serve
REM     (leave that terminal open)
REM  3. Then run this bat from a second terminal.
REM
REM  Why: without OLLAMA_NUM_PARALLEL=1, Ollama may dispatch multiple
REM  concurrent generation requests to a model that just crashed its
REM  runner, compounding the recovery time. Without OLLAMA_MAX_LOADED_MODELS=2,
REM  the embedder and chat model fight over load slots and cause churn.
REM
REM  Fortifications in code (as of 2026-05-18):
REM   - OllamaLLM: 180 s per-call timeout (prevents 8-17 min hung chats)
REM   - OllamaLLM: num_predict=512 (caps runaway generation that drove BSODs)
REM   - experiment_utils: 60 s recovery wait + Ollama health poll after
REM     "model runner has unexpectedly stopped" before retrying
REM   - keep_alive=30m on both LLM and embedder calls
REM   - 30 s cooldown between configs, 0.5 s row pace, 10 s rest every 50 rows
REM ============================================================

"D:\Projects\2026Project\.venv\Scripts\python.exe" -u ^
  "D:\Projects\2026Project\scripts\run_experiment_1.py" ^
  --resume --max-cost 30 --no-gallery ^
  >> "D:\Projects\2026Project\results\experiment_1_run.log" 2>&1
