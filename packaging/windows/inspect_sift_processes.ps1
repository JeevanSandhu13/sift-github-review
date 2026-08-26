param(
    [string]$Output = "C:\Users\Public\sift-processes.json"
)

$Processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "Sift*" -or $_.Name -like "unins*" -or
    $_.Name -eq "ISCC.exe"
} | ForEach-Object {
    [ordered]@{
        process_id = $_.ProcessId
        parent_process_id = $_.ParentProcessId
        name = $_.Name
        creation_date = [string]$_.CreationDate
        kernel_time = $_.KernelModeTime
        user_time = $_.UserModeTime
        command_line = $_.CommandLine
    }
})
ConvertTo-Json -InputObject $Processes -Depth 3 |
    Set-Content -LiteralPath $Output -Encoding UTF8
