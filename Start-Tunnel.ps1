param(
    [ValidateNotNullOrEmpty()]
    [string[]]$AllowedEmails = @('minhnvq2.work@gmail.com', 'lambach35899@gmail.com'),
    [ValidateRange(1, 65535)]
    [int]$Port = 3000
)

$ErrorActionPreference = 'Stop'
$originUrl = "http://127.0.0.1:$Port"
$tunnelFolder = Join-Path $PSScriptRoot '.runtime\cloudflare'
$localConnector = Join-Path $tunnelFolder 'cloudflared.exe'

# Access validates token timestamps before forwarding requests to AnnoPilot.
# A disabled clock service caused an observed "token issued in the future (iat)" failure.
try {
    $timeService = Get-Service -Name W32Time -ErrorAction Stop
    if ($timeService.StartType -eq 'Disabled') {
        Write-Warning 'Windows Time dang bi tat. Neu dang nhap bao "token issued in the future (iat)", mo PowerShell bang Run as administrator va chay:'
        Write-Host 'Set-Service -Name W32Time -StartupType Automatic'
        Write-Host 'Start-Service -Name W32Time'
        Write-Host 'w32tm /resync /rediscover'
    }
} catch {
    Write-Verbose "Could not inspect Windows Time service: $($_.Exception.Message)"
}

# A Quick Tunnel exposes the same dataset/training API as the local application.
# Always require email verification rather than silently publishing an open admin UI.
$emailList = @($AllowedEmails | ForEach-Object { $_.Split(',') } |
    ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique)
if (!$emailList.Count -or ($emailList | Where-Object { $_ -notmatch '^[^\s@*,]+@[^\s@*,]+\.[^\s@*,]+$' })) {
    throw 'Provide explicit email addresses, for example: .\Start-Tunnel.ps1 -AllowedEmails "you@example.com","reviewer@example.com"'
}

try {
    $tunnelHealth = Invoke-RestMethod -Uri "$originUrl/api/health" -TimeoutSec 10
    if ($tunnelHealth.application -ne 'AnnoPilot-ModelQC' -or $tunnelHealth.status -ne 'ok') {
        throw 'The selected port is not serving a healthy AnnoPilot application.'
    }
    $frontendResponse = Invoke-WebRequest -Uri $originUrl -UseBasicParsing -TimeoutSec 10
    if ($frontendResponse.StatusCode -ne 200) { throw 'Frontend did not return HTTP 200.' }
} catch {
    throw "AnnoPilot is not ready at $originUrl. Start frontend and backend first. $($_.Exception.Message)"
}

$installedConnector = Get-Command cloudflared.exe -ErrorAction SilentlyContinue
if ($installedConnector) {
    $connectorPath = $installedConnector.Source
} elseif (Test-Path -LiteralPath $localConnector) {
    $connectorPath = $localConnector
} else {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        throw 'This installer targets Windows x64. Install a connector matching your architecture from the official Cloudflare downloads page.'
    }
    New-Item -ItemType Directory -Path $tunnelFolder -Force | Out-Null
    $partialConnector = Join-Path $tunnelFolder 'cloudflared-download.part'
    Write-Host 'Downloading cloudflared from the official Cloudflare GitHub release...'
    try {
        Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' `
            -OutFile $partialConnector -UseBasicParsing -TimeoutSec 180
        if ((Get-Item -LiteralPath $partialConnector).Length -lt 1000000) {
            throw 'The downloaded file is too small to be a valid connector.'
        }
        Move-Item -LiteralPath $partialConnector -Destination $localConnector -Force
    } catch {
        throw "Could not download cloudflared. Check Internet access to github.com and retry this script. $($_.Exception.Message)"
    }
    $connectorPath = $localConnector
}

& $connectorPath --version
if ($LASTEXITCODE -ne 0) { throw 'cloudflared could not start. Remove the invalid connector and download it again from Cloudflare.' }
$connectorHelp = & $connectorPath tunnel --help 2>&1 | Out-String
if ($connectorHelp -notmatch '\-\-allowed-mail') {
    throw 'This cloudflared version does not support Quick Tunnel email protection (--allowed-mail). Update cloudflared or configure a named Tunnel with Cloudflare Access before publishing AnnoPilot.'
}
if ($connectorHelp -notmatch '\-\-http-host-header') {
    throw 'This cloudflared version does not expose --http-host-header. Update the connector before using the Vite origin.'
}

Write-Host "Local application is ready at $originUrl"
Write-Host 'The HTTPS test URL will appear below. Share it only with the allowed reviewers.'
Write-Host 'Visitors must verify an allowed email address using the one-time PIN.'
Write-Host 'Keep this terminal open. Press Ctrl+C to stop the public link; local AnnoPilot remains running.'

# Preserve Vite host validation: rewrite the origin Host instead of allowing all public hosts.
$connectorArgs = @('tunnel', '--url', $originUrl, '--http-host-header', "localhost:$Port")
foreach ($allowedEmail in $emailList) { $connectorArgs += @('--allowed-mail', $allowedEmail) }
& $connectorPath @connectorArgs
if ($LASTEXITCODE -ne 0) { throw "Tunnel exited with code $LASTEXITCODE. Review the cloudflared message above." }
