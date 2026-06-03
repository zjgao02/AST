param(
    [int]$MaxProteinsPerFamily = 200,
    [int]$PageSize = 100,
    [double]$SleepSeconds = 0.05,
    [int]$EmbedBatchSize = 16
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LogDir = Join-Path $Root "artifacts\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BuildLog = Join-Path $LogDir "atf_kb_representative_build_$Stamp.log"
$EmbedLog = Join-Path $LogDir "atf_kb_representative_embed_$Stamp.log"
$StatusPath = Join-Path $LogDir "atf_kb_representative_latest_status.json"
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
        max_proteins_per_family = $MaxProteinsPerFamily
        build_log = $BuildLog
        embed_log = $EmbedLog
    }
    $payload | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}

Push-Location $Root
try {
    Write-Status -Phase "build" -Status "running" -Message "Downloading bounded representative ATF InterPro records."
    "[$(Get-Date -Format o)] Starting representative ATF KB build; max_proteins_per_family=$MaxProteinsPerFamily" |
        Tee-Object -FilePath $BuildLog
    $buildOutput = & $Conda run -n pytorch python data\scripts\build_atf_interpro_kb_v0.py `
        --max-proteins-per-family $MaxProteinsPerFamily `
        --page-size $PageSize `
        --sleep $SleepSeconds `
        --include-full `
        --progress-every 50 2>&1
    $buildExitCode = $LASTEXITCODE
    $buildOutput | Tee-Object -Append -FilePath $BuildLog
    if ($buildExitCode -ne 0) {
        throw "Representative ATF KB build failed with exit code $buildExitCode"
    }

    Write-Status -Phase "embed" -Status "running" -Message "Embedding bounded representative ATF InterPro records."
    "[$(Get-Date -Format o)] Starting representative ATF KB embedding; batch_size=$EmbedBatchSize" |
        Tee-Object -FilePath $EmbedLog
    $embedOutput = & $Conda run -n pytorch python data\scripts\embed_atf_interpro_records_v0.py `
        --batch-size $EmbedBatchSize 2>&1
    $embedExitCode = $LASTEXITCODE
    $embedOutput | Tee-Object -Append -FilePath $EmbedLog
    if ($embedExitCode -ne 0) {
        throw "Representative ATF KB embedding failed with exit code $embedExitCode"
    }

    Write-Status -Phase "complete" -Status "complete" -Message "Representative ATF KB build and embedding completed."
}
catch {
    Write-Status -Phase "failed" -Status "failed" -Message $_.Exception.Message
    throw
}
finally {
    Pop-Location
}
