param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PingUrl,

    [Parameter(Mandatory = $false, Position = 1)]
    [string]$RepoDir = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("HH:mm:ss")
    Write-Host "[$stamp] $Message"
}

function Write-Pass {
    param([string]$Message)
    Write-Log "PASSED -- $Message"
}

function Write-Fail {
    param([string]$Message)
    Write-Log "FAILED -- $Message"
}

try {
    $repoPath = (Resolve-Path -Path $RepoDir).Path
}
catch {
    Write-Fail "Directory '$RepoDir' was not found"
    exit 1
}

$PingUrl = $PingUrl.TrimEnd("/")

Write-Host ""
Write-Host "========================================"
Write-Host "  OpenEnv Submission Validator (PowerShell)"
Write-Host "========================================"
Write-Log "Repo:     $repoPath"
Write-Log "Ping URL: $PingUrl"
Write-Host ""

Write-Log "Step 1/3: Pinging HF Space ($PingUrl/reset) ..."
try {
    $invokeParams = @{
        Method = "Post"
        Uri = "$PingUrl/reset"
        ContentType = "application/json"
        Body = "{}"
        TimeoutSec = 30
    }
    $response = Invoke-WebRequest @invokeParams

    if ($response.StatusCode -ne 200) {
        Write-Fail "HF Space /reset returned HTTP $($response.StatusCode) (expected 200)"
        exit 1
    }

    Write-Pass "HF Space is live and responds to /reset"
}
catch {
    Write-Fail "HF Space not reachable or returned an error: $($_.Exception.Message)"
    exit 1
}

Write-Log "Step 2/3: Running docker build ..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail "docker command not found. Install Docker Desktop first."
    exit 1
}

$dockerContext = $null
if (Test-Path (Join-Path $repoPath "Dockerfile")) {
    $dockerContext = $repoPath
}
elseif (Test-Path (Join-Path $repoPath "server/Dockerfile")) {
    $dockerContext = (Join-Path $repoPath "server")
}
else {
    Write-Fail "No Dockerfile found in repo root or server/ directory"
    exit 1
}

Write-Log "  Found Dockerfile in $dockerContext"
$buildOutput = & docker build $dockerContext 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Docker build failed"
    $buildOutput | Select-Object -Last 25 | ForEach-Object { Write-Host $_ }
    exit 1
}
Write-Pass "Docker build succeeded"

Write-Log "Step 3/3: Running openenv validate ..."
if (-not (Get-Command openenv -ErrorAction SilentlyContinue)) {
    Write-Fail "openenv command not found. Install with: pip install openenv-core"
    exit 1
}

Push-Location $repoPath
try {
    $validateOutput = & openenv validate 2>&1
    $validateExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($validateExitCode -ne 0) {
    Write-Fail "openenv validate failed"
    $validateOutput | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Pass "openenv validate passed"
if ($validateOutput) {
    $validateOutput | ForEach-Object { Write-Log "  $_" }
}

Write-Host ""
Write-Host "========================================"
Write-Host "  All 3/3 checks passed"
Write-Host "  Your submission is ready to submit"
Write-Host "========================================"
Write-Host ""

exit 0
