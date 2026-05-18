$since = (Get-Date).AddHours(-24)
Write-Host "=== All Kernel-Power 41 and EventLog 6008 (unexpected shutdowns) in last 24h ==="
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since} -ErrorAction SilentlyContinue |
    Where-Object { ($_.Id -eq 41 -and $_.ProviderName -eq 'Microsoft-Windows-Kernel-Power') -or
                   ($_.Id -eq 6008 -and $_.ProviderName -eq 'EventLog') -or
                   ($_.Id -eq 1001 -and $_.ProviderName -eq 'Microsoft-Windows-WER-SystemErrorReporting') -or
                   ($_.Id -eq 12  -and $_.ProviderName -eq 'Microsoft-Windows-Kernel-General') -or
                   ($_.Id -eq 13  -and $_.ProviderName -eq 'Microsoft-Windows-Kernel-General') } |
    Sort-Object TimeCreated |
    Select-Object TimeCreated, Id, ProviderName,
        @{n='Msg';e={ ($_.Message -split "`n")[0] }} |
    Format-Table -AutoSize -Wrap

Write-Host ""
Write-Host "=== Recent minidumps ==="
Get-ChildItem C:\WINDOWS\Minidump -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name, LastWriteTime, Length

Write-Host ""
Write-Host "=== Last 10 events around 02:14 - 02:45 (any provider, level 1-3) ==="
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime='2026-05-18 02:00:00'; EndTime='2026-05-18 03:00:00'; Level=1,2,3} -ErrorAction SilentlyContinue |
    Sort-Object TimeCreated |
    Select-Object TimeCreated, Id, LevelDisplayName, ProviderName,
        @{n='Msg';e={ ($_.Message -split "`n")[0].Substring(0, [Math]::Min(80, ($_.Message -split "`n")[0].Length)) }} |
    Format-Table -AutoSize
