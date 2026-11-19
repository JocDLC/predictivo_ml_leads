function Get-HarnessObservedVersion {
    param([string]$Command, [string[]]$Arguments = @('--version'), [string]$Pattern,
        [int]$TimeoutMilliseconds = 10000)
    $resolved = Get-Command $Command -CommandType Application, ExternalScript -ErrorAction Stop | Select-Object -First 1
    $literalArgs = @($Arguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ','
    $scriptText = "& '" + $resolved.Source.Replace("'", "''") + "' @(" + $literalArgs + "); if (`$LASTEXITCODE -ne 0) { exit 1 }"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($scriptText))
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = (Get-Process -Id $PID).Path
    $start.Arguments = "-NoProfile -NonInteractive -EncodedCommand $encoded"
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    $timer = [Diagnostics.Stopwatch]::StartNew()
    try {
        $null = $process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            & taskkill.exe /PID $process.Id /T /F *> $null
            throw 'Timeout de sonda de version.'
        }
        if ($process.ExitCode -ne 0) { throw 'Sonda de version termino con error.' }
        $remaining = [math]::Max(1, $TimeoutMilliseconds - [int]$timer.ElapsedMilliseconds)
        if (-not [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdout, $stderr), $remaining)) {
            throw 'Timeout leyendo salida de version.'
        }
        $text = $stdout.GetAwaiter().GetResult().Trim()
        $null = $stderr.GetAwaiter().GetResult()
        if ($text -notmatch $Pattern) { throw 'Version no interpretable.' }
        return [version]$Matches[1]
    }
    finally { $process.Dispose() }
}

function Test-HarnessVersionPolicy {
    param([version]$Actual, [version]$Expected, [ValidateSet('minimum', 'exact')][string]$Policy)
    if ($Policy -eq 'minimum') { return $Actual -ge $Expected }
    return $Actual -eq $Expected
}
