$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LogDir = Join-Path $Root "artifacts\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BuildLog = Join-Path $LogDir "atf_kb_full_build_$Stamp.log"
$EmbedLog = Join-Path $LogDir "atf_kb_full_embed_$Stamp.log"
$StatusPath = Join-Path $LogDir "atf_kb_full_latest_status.json"
$Conda = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { "conda" }

function Write-Status {
    param(
        [string]$Phase,
        [string]$Status,
        [string]$Message = ""
    )
    $payload = [ordered]@{
        phase = $Phase
        status = $Status
        message = $Message
        updated_at = (Get-Date).ToString("o")
        build_log = $BuildLog
        embed_log = $EmbedLog
    }
    $payload | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}

Push-Location $Root
try {
    Write-Status -Phase "build" -Status "running" -Message "Downloading full ATF InterPro records."
    "[$(Get-Date -Format o)] Starting full ATF KB build" | Tee-Object -FilePath $BuildLog
    & $Conda run -n pytorch python data\scripts\build_atf_interpro_kb_v0.py `
        --max-proteins-per-family 0 `
        --page-size 100 `
        --sleep 0.2 `
        --include-full `
        --progress-every 1000 2>&1 | Tee-Object -Append -FilePath $BuildLog
    if ($LASTEXITCODE -ne 0) {
        throw "ATF KB build failed with exit code $LASTEXITCODE"
    }

    Write-Status -Phase "embed" -Status "running" -Message "Embedding full ATF InterPro records."
    "[$(Get-Date -Format o)] Starting full ATF KB embedding" | Tee-Object -FilePath $EmbedLog
    & $Conda run -n pytorch python data\scripts\embed_atf_interpro_records_v0.py --batch-size 16 2>&1 |
        Tee-Object -Append -FilePath $EmbedLog
    if ($LASTEXITCODE -ne 0) {
        throw "ATF KB embedding failed with exit code $LASTEXITCODE"
    }

    Write-Status -Phase "complete" -Status "complete" -Message "Full ATF KB build and embedding completed."
}
catch {
    Write-Status -Phase "failed" -Status "failed" -Message $_.Exception.Message
    throw
}
finally {
    Pop-Location
}
