param(
    [string]$Output = "C:\Users\Public\sift-native-release-task.json"
)

$ErrorActionPreference = "Stop"
$TaskName = "Sift Native Release Build"
$Task = Get-ScheduledTask -TaskName $TaskName
$Info = Get-ScheduledTaskInfo -TaskName $TaskName
[ordered]@{
    task_name = $TaskName
    state = [string]$Task.State
    last_run_time = [string]$Info.LastRunTime
    last_task_result = $Info.LastTaskResult
    next_run_time = [string]$Info.NextRunTime
    missed_runs = $Info.NumberOfMissedRuns
} | ConvertTo-Json | Set-Content -LiteralPath $Output -Encoding UTF8
