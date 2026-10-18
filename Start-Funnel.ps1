param(
    [ValidateRange(1, 65535)]
    [int]$Port = 3000
)

$ErrorActionPreference = 'Stop'
$taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskPrincipal = New-Object Security.Principal.WindowsPrincipal($taskIdentity)
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Mo PowerShell bang Run as administrator, cd vao thu muc project, roi chay lai .\Start-Funnel.ps1. Dich vu Tailscale tren may nay yeu cau quyen Administrator.'
}

$tailscaleCommand = Get-Command tailscale.exe -ErrorAction SilentlyContinue
if ($tailscaleCommand) {
    $tailscalePath = $tailscaleCommand.Source
} else {
    $tailscalePath = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
}
if (-not (Test-Path -LiteralPath $tailscalePath)) {
    throw 'Chua tim thay Tailscale. Cai tu https://tailscale.com/download/windows va dang nhap tai khoan cua ban truoc.'
}

$originUrl = "http://127.0.0.1:$Port"
try {
    $health = Invoke-RestMethod -Uri "$originUrl/api/health" -TimeoutSec 10
    if ($health.status -ne 'ok' -or $health.application -ne 'AnnoPilot-ModelQC') {
        throw 'API tra ve khong phai AnnoPilot.'
    }
    $frontend = Invoke-WebRequest -Uri $originUrl -UseBasicParsing -TimeoutSec 10
    if ($frontend.StatusCode -ne 200) { throw 'Frontend chua san sang.' }
} catch {
    throw "AnnoPilot chua san sang tai $originUrl. Hay chay backend va frontend truoc. $($_.Exception.Message)"
}

Write-Host 'Dang mo AnnoPilot qua Tailscale Funnel. Nguoi xem chi can trinh duyet, khong can cai Tailscale.'
Write-Host 'Day la link cong khai: danh sach email cua Cloudflare khong duoc ap dung. Nguoi co link co the truy cap cac chuc nang hien tai cua web.'
Write-Host 'Neu Tailscale in link yeu cau bat Funnel/HTTPS, mo link do bang tai khoan cua ban, hoan tat va chay lai script.'

& $tailscalePath funnel --bg --yes --https=443 $originUrl
if ($LASTEXITCODE -ne 0) {
    throw 'Chua bat duoc Funnel. Xem thong bao Tailscale phia tren; kiem tra dang nhap, quyen truy cap va trang bat Funnel.'
}

& $tailscalePath funnel status
if ($LASTEXITCODE -ne 0) { throw 'Khong doc duoc trang thai Funnel. Xem thong bao phia tren.' }

Write-Host 'Gui dia chi HTTPS duoc Tailscale in o tren cho nguoi dung.'
Write-Host 'Funnel dang chay nen. Van can giu may chu, Tailscale, frontend va backend hoat dong.'
Write-Host 'Dung chia se: & "C:\Program Files\Tailscale\tailscale.exe" funnel --https=443 off'
