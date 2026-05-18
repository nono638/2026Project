# Stop any running Ollama and restart `ollama serve` with env vars tuned
# for sustained batch workloads on the 5090 Laptop GPU.
#
# Why this exists:
# - The Ollama tray app starts the server with default env (MAX_LOADED_MODELS=3,
#   NUM_PARALLEL ~= auto). On the 5090 we've correlated VRAM allocator churn
#   from concurrent residency / eviction with repeated BSODs
#   (0x133, 0x3b, 0x0a, 0x116 all today, 2026-05-18).
# - Env vars set in any shell AFTER the tray app launched are ignored by the
#   already-running server. The only way to apply them is to kill the existing
#   processes and re-launch `ollama serve` with the env populated.
#
# Run from PowerShell (not admin needed):
#   powershell -ExecutionPolicy Bypass -File scripts/start_ollama_safe.ps1
#
# Side effect: Ollama is restarted. Anything currently using it will fail.

$ErrorActionPreference = 'Stop'

Write-Host "=== Stopping any running Ollama processes ==="
$names = @('ollama app', 'ollama', 'ollama_llama_server')
foreach ($n in $names) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  killing $($_.ProcessName) (pid $($_.Id))"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2

# 5090-stability env vars — see Tier-2 of the research note in the
# memory file project_5090_stability.md.
# - MAX_LOADED_MODELS=1 forces clean unload between embedder<->chat instead of
#   keeping both resident; the concurrent residency is the VRAM-churn regime
#   that has correlated with every BSOD this week.
# - NUM_PARALLEL=1: our pipeline is already serial, this just locks the server
#   into the matching mode and disables Ollama's per-slot KV duplication.
# - KEEP_ALIVE=24h: well past any single experiment config; combined with
#   MAX_LOADED_MODELS=1 it still evicts on model-switch (Ollama's eviction
#   rule supersedes keep_alive when slots are full).
# - MAX_QUEUE=512: default is 512; explicit so it survives a future default change.
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NUM_PARALLEL      = '1'
$env:OLLAMA_KEEP_ALIVE        = '24h'
$env:OLLAMA_MAX_QUEUE         = '512'
# OLLAMA_FLASH_ATTENTION and OLLAMA_KV_CACHE_TYPE intentionally NOT set:
# Ollama auto-disables FA when an embed model is loaded (mixed pipeline),
# and q8_0 KV silently falls back to f16 on archs that don't support it.

$logPath = Join-Path $PSScriptRoot '..\results\ollama_serve.log'
$logPath = [System.IO.Path]::GetFullPath($logPath)
$logDir  = Split-Path $logPath -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

Write-Host ""
Write-Host "=== Starting `ollama serve` with stability env ==="
Write-Host "  OLLAMA_MAX_LOADED_MODELS = $env:OLLAMA_MAX_LOADED_MODELS"
Write-Host "  OLLAMA_NUM_PARALLEL      = $env:OLLAMA_NUM_PARALLEL"
Write-Host "  OLLAMA_KEEP_ALIVE        = $env:OLLAMA_KEEP_ALIVE"
Write-Host "  OLLAMA_MAX_QUEUE         = $env:OLLAMA_MAX_QUEUE"
Write-Host "  log: $logPath"

$serveCmd = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $serveCmd) {
    Write-Error "ollama executable not found on PATH."
    exit 1
}

# Launch detached so this shell can exit without killing the server.
# Inherits the env vars set above via Start-Process's default behavior.
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = $serveCmd
$psi.Arguments = 'serve'
$psi.UseShellExecute        = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.CreateNoWindow         = $true
foreach ($k in 'OLLAMA_MAX_LOADED_MODELS','OLLAMA_NUM_PARALLEL','OLLAMA_KEEP_ALIVE','OLLAMA_MAX_QUEUE') {
    $psi.EnvironmentVariables[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
}

$proc = [System.Diagnostics.Process]::Start($psi)
Start-Job -Name "ollama-stdout-$($proc.Id)" -ScriptBlock {
    param($p, $log)
    while (-not $p.HasExited) {
        $line = $p.StandardOutput.ReadLine()
        if ($null -ne $line) { Add-Content -Path $log -Value $line }
    }
} -ArgumentList $proc, $logPath | Out-Null
Start-Job -Name "ollama-stderr-$($proc.Id)" -ScriptBlock {
    param($p, $log)
    while (-not $p.HasExited) {
        $line = $p.StandardError.ReadLine()
        if ($null -ne $line) { Add-Content -Path $log -Value $line }
    }
} -ArgumentList $proc, $logPath | Out-Null

Write-Host ""
Write-Host "Started ollama serve (pid $($proc.Id))."

# Wait for the API to respond before exiting.
$deadline = (Get-Date).AddSeconds(30)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if ($ready) {
    Write-Host "Ollama API is responding at http://127.0.0.1:11434."
} else {
    Write-Warning "Ollama did not respond within 30s. Check $logPath."
    exit 2
}
