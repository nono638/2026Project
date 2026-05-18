$since = (Get-Date).AddHours(-12)
Write-Host "=== Critical/Error events in System log since $since ==="
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since; Level=1,2,3} -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -in 41,6008,1074,1001,219,6005,6006,117 -or $_.ProviderName -match 'Kernel|nvlddmkm|Display|WHEA|BugCheck' } |
    Select-Object TimeCreated, Id, LevelDisplayName, ProviderName,
        @{n='Msg';e={ ($_.Message -split "`n")[0] }} |
    Sort-Object TimeCreated |
    Format-Table -AutoSize -Wrap

Write-Host ""
Write-Host "=== Last boot time ==="
(Get-CimInstance Win32_OperatingSystem).LastBootUpTime

Write-Host ""
Write-Host "=== Any BugCheck (BSOD) events? ==="
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} -ErrorAction SilentlyContinue |
    Format-List TimeCreated, Id, Message

Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$since; ID=1001} -ErrorAction SilentlyContinue |
    Where-Object { $_.ProviderName -match 'BugCheck' } |
    Format-List TimeCreated, Id, ProviderName, Message
