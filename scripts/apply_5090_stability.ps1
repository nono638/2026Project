# Apply system-level 5090 stability mitigations.
#
# REQUIRES ADMINISTRATOR. Run as:
#   Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\scripts\apply_5090_stability.ps1`""
#
# What this does (and why), based on 2026-05-18 research into the recurring
# nvlddmkm BSOD pattern on RTX 5090 Laptop under Ollama workload:
#
# 1) TDR registry: raises Windows' Timeout Detection and Recovery deadline
#    from 2s/5s defaults to 60s. Long LLM kernels can legitimately hold
#    the GPU past the default and trigger 0x116 VIDEO_TDR_ERROR.
#    See: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
#         https://www.pugetsystems.com/labs/hpc/working-around-tdr-in-windows-for-a-better-gpu-computing-experience-777/
#
# 2) Hardware-accelerated GPU scheduling (HAGS): off. Implicated in 5090
#    nvlddmkm DPC_WATCHDOG instability and in the "Ollama unload after
#    couple of minutes" thread on the NVIDIA dev forum.
#    See: https://forums.developer.nvidia.com/t/rtx-5090-total-failure-hang-unload-in-ollama-after-couple-of-minutes/341659
#         https://essenceofcode.com/2025/04/22/hardware-accelerated-gpu-scheduling-instability/
#
# 3) PCIe ASPM: off (high-performance plan + Link State Power Management = off).
#    ASPM transitions on dGPU laptops are a documented trigger for nvlddmkm
#    stalls. Our per-row pacing creates idle windows where ASPM may downshift.
#    See: https://forums.tomshardware.com/threads/bsod-dpc-watchdog-violation-on-windows-11.3781753/
#
# A reboot is REQUIRED for (1) and (2). (3) takes effect immediately.

#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'

function Write-Section($t) {
    Write-Host ""
    Write-Host "=== $t ===" -ForegroundColor Cyan
}

# --- 1) TDR registry ---------------------------------------------------------
Write-Section "TDR registry"
$tdrKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
# TdrDelay=60: GPU may be unresponsive for up to 60s before reset (default 2).
# TdrDdiDelay=60: how long the OS waits for the driver to leave a DDI call (default 5).
# TdrLevel=3: full recovery (default). Do NOT set to 0 (disables detection
# entirely — masks bugs instead of surviving them).
Set-ItemProperty -Path $tdrKey -Name 'TdrDelay'    -Value 60 -Type DWord
Set-ItemProperty -Path $tdrKey -Name 'TdrDdiDelay' -Value 60 -Type DWord
Set-ItemProperty -Path $tdrKey -Name 'TdrLevel'    -Value 3  -Type DWord
Write-Host "TdrDelay=60, TdrDdiDelay=60, TdrLevel=3 set under $tdrKey"

# --- 2) HAGS off -------------------------------------------------------------
Write-Section "Hardware-Accelerated GPU Scheduling"
# HwSchMode: 1 = off, 2 = on. Note: NVIDIA driver re-installs sometimes flip
# this back to 2 — check after any driver update.
Set-ItemProperty -Path $tdrKey -Name 'HwSchMode' -Value 1 -Type DWord
Write-Host "HwSchMode=1 (HAGS off) set."

# --- 3) Power plan + ASPM ----------------------------------------------------
Write-Section "Power plan + PCIe ASPM"
# Set High Performance plan (GUID 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c).
& powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
# PCI Express > Link State Power Management — subgroup 501a4d13... setting ee12f906...
# 0 = Off (max performance), 1 = Moderate, 2 = Maximum savings.
& powercfg /setacvalueindex SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0
& powercfg /setdcvalueindex SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0
& powercfg /setactive SCHEME_CURRENT
Write-Host "Power plan = High performance; PCIe Link State Power Management = Off (AC + DC)."

# --- Summary -----------------------------------------------------------------
Write-Section "Done"
Write-Host "REBOOT REQUIRED for TDR and HAGS changes to take effect." -ForegroundColor Yellow
Write-Host "After reboot:"
Write-Host "  1. Run scripts/start_ollama_safe.ps1 (regular PowerShell, not admin)."
Write-Host "  2. Activate venv and resume: python scripts/run_experiment_1.py --resume"
Write-Host ""
Write-Host "If BSODs persist after this, consider rolling back to NVIDIA Studio 591.59"
Write-Host "via DDU in Safe Mode -- last driver branch with widely-reported 50-series"
Write-Host "stability. See project_5090_stability.md for details."
