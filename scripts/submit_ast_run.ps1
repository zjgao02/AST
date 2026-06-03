param(
    [ValidateSet("tetr_dopamine", "cd25_scfv")]
    [string]$Case = "tetr_dopamine",

    [ValidateSet("assets", "preview", "inner-smoke", "outer", "formal", "all")]
    [string]$Stage = "preview",

    [ValidateSet("smoke", "cheap", "formal")]
    [string]$Profile = "smoke",

    [int]$OuterIterations = 0,
    [int]$InnerIterations = 0,

    [ValidateSet("auto", "on", "off")]
    [string]$Protenix = "auto",

    [ValidateSet("auto", "on", "off")]
    [string]$ExternalKb = "auto",

    [ValidateSet("auto", "on", "off")]
    [string]$ExternalRetrieval = "auto",

    [double]$ProgenWeight = -1.0,

    [string]$RunName = "",
    [string]$CondaEnv = "pytorch",
    [string]$CondaExe = "conda",
    [switch]$NoConda,
    [switch]$DryRun,
    [switch]$SkipLlmKeyCheck
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Get-CaseDefault {
    param([string]$CaseId)
    if ($CaseId -eq "tetr_dopamine") {
        return @{
            InnerIterations = 240
            ProgenWeight = 0.5
            ExternalRetrieval = $false
        }
    }
    return @{
        InnerIterations = 1200
        ProgenWeight = 1.0
        ExternalRetrieval = $true
    }
}

function Get-ProfileDefault {
    param([string]$ProfileName, [hashtable]$CaseDefault)
    if ($ProfileName -eq "smoke") {
        return @{
            OuterIterations = 1
            InnerIterations = 1
            Protenix = $false
            ExternalKb = $false
            ExternalRetrieval = $false
            ProgenWeight = 0.0
        }
    }
    if ($ProfileName -eq "cheap") {
        return @{
            OuterIterations = 10
            InnerIterations = 10
            Protenix = $false
            ExternalKb = $true
            ExternalRetrieval = [bool]$CaseDefault.ExternalRetrieval
            ProgenWeight = [double]$CaseDefault.ProgenWeight
        }
    }
    return @{
        OuterIterations = 200
        InnerIterations = [int]$CaseDefault.InnerIterations
        Protenix = $true
        ExternalKb = $true
        ExternalRetrieval = [bool]$CaseDefault.ExternalRetrieval
        ProgenWeight = [double]$CaseDefault.ProgenWeight
    }
}

function Resolve-Toggle {
    param([string]$Value, [bool]$Default)
    if ($Value -eq "on") {
        return $true
    }
    if ($Value -eq "off") {
        return $false
    }
    return $Default
}

function Format-Command {
    param([string[]]$Command)
    return (($Command | ForEach-Object {
        if ($_ -match "\s") {
            '"' + $_ + '"'
        } else {
            $_
        }
    }) -join " ")
}

function Invoke-CommandLine {
    param([string[]]$Command)
    Write-Host ("+ " + (Format-Command $Command))
    if ($DryRun) {
        return
    }
    $cmdArgs = @()
    if ($Command.Count -gt 1) {
        $cmdArgs = $Command[1..($Command.Count - 1)]
    }
    & $Command[0] @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Python {
    param([string[]]$PythonArgs)
    if ($NoConda) {
        Invoke-CommandLine (@("python") + $PythonArgs)
    } else {
        Invoke-CommandLine (@($CondaExe, "run", "-n", $CondaEnv, "python") + $PythonArgs)
    }
}

$caseDefault = Get-CaseDefault $Case
$profileDefault = Get-ProfileDefault $Profile $caseDefault

if ($OuterIterations -le 0) {
    $OuterIterations = [int]$profileDefault.OuterIterations
}
if ($InnerIterations -le 0) {
    $InnerIterations = [int]$profileDefault.InnerIterations
}
if ($ProgenWeight -lt 0.0) {
    $ProgenWeight = [double]$profileDefault.ProgenWeight
}
$useProtenix = Resolve-Toggle $Protenix ([bool]$profileDefault.Protenix)
$useExternalKb = Resolve-Toggle $ExternalKb ([bool]$profileDefault.ExternalKb)
$useExternalRetrieval = Resolve-Toggle $ExternalRetrieval ([bool]$profileDefault.ExternalRetrieval)

if (-not $RunName) {
    $RunName = ("{0}_{1}_{2}" -f $Case, $Profile, (Get-Date -Format "yyyyMMdd_HHmmss"))
}
if (-not $env:ASTEVOLVE_PROJECT_ROOT) {
    $env:ASTEVOLVE_PROJECT_ROOT = [string]$ProjectRoot
}
if (-not $env:ASTEVOLVE_DATA_ROOT) {
    $env:ASTEVOLVE_DATA_ROOT = Join-Path $ProjectRoot "data"
}
if (-not $env:ASTEVOLVE_ARTIFACT_ROOT) {
    $env:ASTEVOLVE_ARTIFACT_ROOT = Join-Path $ProjectRoot "artifacts"
}
if (-not $env:ASTEVOLVE_TMP_ROOT) {
    $env:ASTEVOLVE_TMP_ROOT = Join-Path $env:ASTEVOLVE_ARTIFACT_ROOT "tmp"
}

$runRoot = Join-Path $env:ASTEVOLVE_ARTIFACT_ROOT ("runs\{0}\{1}" -f $Case, $RunName)
$env:ASTEVOLVE_CASE_ID = $Case
$env:ASTEVOLVE_INNER_ITERATIONS = [string]$InnerIterations
$env:ASTEVOLVE_ENABLE_PROTENIX = if ($useProtenix) { "1" } else { "0" }
$env:ASTEVOLVE_ENABLE_EXTERNAL_KB = if ($useExternalKb) { "1" } else { "0" }
$env:ASTEVOLVE_ENABLE_EXTERNAL_RETRIEVAL = if ($useExternalRetrieval) { "1" } else { "0" }
$env:ASTEVOLVE_PROGEN_WEIGHT = [string]$ProgenWeight
$env:ASTEVOLVE_MCTS_OUTPUT_DIR = Join-Path $runRoot "inner"
$env:ASTEVOLVE_PROTENIX_TMP = Join-Path $runRoot "protenix_tmp"
$env:ASTEVOLVE_PROTENIX_NUM_WORKERS = if ($env:ASTEVOLVE_PROTENIX_NUM_WORKERS) { $env:ASTEVOLVE_PROTENIX_NUM_WORKERS } else { "1" }
$env:ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA = if ($env:ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA) { $env:ASTEVOLVE_PROTENIX_COMPLEX_USE_MSA } else { "0" }
$env:ASTEVOLVE_PROTENIX_COMPLEX_CYCLE = if ($env:ASTEVOLVE_PROTENIX_COMPLEX_CYCLE) { $env:ASTEVOLVE_PROTENIX_COMPLEX_CYCLE } else { "1" }
$env:ASTEVOLVE_PROTENIX_COMPLEX_STEP = if ($env:ASTEVOLVE_PROTENIX_COMPLEX_STEP) { $env:ASTEVOLVE_PROTENIX_COMPLEX_STEP } else { "1" }
$env:ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE = if ($env:ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE) { $env:ASTEVOLVE_PROTENIX_COMPLEX_SAMPLE } else { "1" }
$env:ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS = if ($env:ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS) { $env:ASTEVOLVE_PROTENIX_COMPLEX_USE_DEFAULT_PARAMS } else { "0" }

Write-Host ("case={0} stage={1} profile={2} run={3}" -f $Case, $Stage, $Profile, $RunName)
Write-Host ("outer_iterations={0} inner_iterations={1} protenix={2} external_kb={3} retrieval={4} progen_weight={5}" -f $OuterIterations, $InnerIterations, $useProtenix, $useExternalKb, $useExternalRetrieval, $ProgenWeight)
Write-Host ("run_root={0}" -f $runRoot)

$steps = @($Stage)
if ($Stage -eq "formal" -or $Stage -eq "all") {
    $steps = @("assets", "preview", "inner-smoke", "outer")
}

foreach ($step in $steps) {
    if ($step -eq "assets") {
        Invoke-Python @("scripts\check_assets.py", "--case", $Case)
    } elseif ($step -eq "preview") {
        Invoke-Python @("cases\$Case\initial_program.py")
    } elseif ($step -eq "inner-smoke") {
        $smokeArgs = @("scripts\smoke_case.py", "--case", $Case, "--iterations", [string]$InnerIterations, "--output-name", $RunName, "--progen-weight", [string]$ProgenWeight, "--json")
        if ($useProtenix) {
            $smokeArgs += "--protenix"
        }
        if ($useExternalKb) {
            $smokeArgs += "--external-kb"
        }
        if ($useExternalRetrieval) {
            $smokeArgs += "--external-retrieval"
        } else {
            $smokeArgs += "--no-external-retrieval"
        }
        Invoke-Python $smokeArgs
    } elseif ($step -eq "outer") {
        if (-not $SkipLlmKeyCheck -and -not $env:ASTEVOLVE_LLM_API_KEY) {
            throw "ASTEVOLVE_LLM_API_KEY is required for outer-loop OpenEvolve runs. Set it or pass -SkipLlmKeyCheck."
        }
        Invoke-Python @(
            "openevolve\openevolve-run.py",
            "cases\$Case\initial_program.py",
            "evaluator.py",
            "--config",
            "cases\$Case\config.yaml",
            "--iterations",
            [string]$OuterIterations,
            "--output",
            (Join-Path $runRoot "openevolve")
        )
    }
}
