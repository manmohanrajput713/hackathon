$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:FIELD_PYTHON = Join-Path $root ".venv/Scripts/python.exe"
$api = Join-Path $root "target/debug/field-console.exe"
$web = Join-Path $root "web"
$logs = Join-Path $root "runtime"
foreach ($file in @($env:FIELD_PYTHON, $api, "$web/.next/BUILD_ID", "$logs/crm.json", "$logs/pump.json", "$logs/curves.json")) {
    if (!(Test-Path -LiteralPath $file)) { throw "Missing $file. Complete the setup in README.md." }
}
function Test-LocalPort([int]$port) {
    $client = [Net.Sockets.TcpClient]::new()
    try { $client.Connect("127.0.0.1",$port); return $true } catch { return $false } finally { $client.Dispose() }
}
if ((Test-LocalPort 8787) -or (Test-LocalPort 3210)) { throw "Port 8787 or 3210 is already in use. Check the running dashboard at http://127.0.0.1:3210." }
$server = Start-Process -FilePath $api -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logs/api.log" -RedirectStandardError "$logs/api-errors.log"
try {
    $next = Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList @("node_modules/next/dist/bin/next","start","-H","127.0.0.1","-p","3210") -WorkingDirectory $web -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logs/web.log" -RedirectStandardError "$logs/web-errors.log"
    for ($i=0; $i -lt 90; $i++) {
        if ($server.HasExited -or $next.HasExited) { throw "A service exited. See runtime logs." }
        try {
            $state=Invoke-RestMethod -Uri "http://127.0.0.1:3210/api/overview" -TimeoutSec 3
            if ($state.wells.Count -gt 0) {
                Write-Output "Dashboard http://127.0.0.1:3210 (API PID $($server.Id), web PID $($next.Id))"
                return
            }
        } catch {}
        Start-Sleep -Milliseconds 750
    }
    throw "Dashboard startup timed out. See runtime logs."
} catch {
    if ($next -and !$next.HasExited) { $next.Kill() }
    if (!$server.HasExited) { $server.Kill() }
    throw
}

