# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonVersion,
    [Parameter(Mandatory = $true)]
    [ValidateSet('modern', 'llvm7')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'

$FullVersions = @{
    '3.11'  = '3.11.11'
    '3.12'  = '3.12.10'
    '3.13'  = '3.13.5'
    '3.14'  = '3.14.3'
    '3.14t' = '3.14.3'
}

function Resolve-FullPythonVersion {
    param([string]$Spec)
    $key = $Spec.Trim()
    if ($FullVersions.ContainsKey($key)) {
        return $FullVersions[$key]
    }
    $base = $key.TrimEnd('t').Trim()
    if ($FullVersions.ContainsKey($base)) {
        return $FullVersions[$base]
    }
    throw "No pinned full Python version for '$Spec'. Update `$FullVersions in ci/windows-llvm-container-build.ps1."
}

function Install-PythonFromPythonOrg {
    param(
        [string]$Spec,
        [string]$FullVer,
        [string]$TargetDir
    )
    $freethreaded = $Spec -match 't$'
    $majMin = [version]($Spec.TrimEnd('t').Trim())
    $installer = Join-Path $env:TEMP "python-$FullVer-amd64.exe"
    $url = "https://www.python.org/ftp/python/$FullVer/python-$FullVer-amd64.exe"
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing

    $argList = @(
        '/quiet',
        'InstallAllUsers=0',
        'SimpleInstall=1',
        'Include_test=0',
        'PrependPath=0',
        "TargetDir=$TargetDir",
        'Include_pip=1',
        'Include_doc=0',
        'Include_lib=1',
        'Include_tcltk=0'
    )
    if ($freethreaded -and $majMin -ge [version]'3.13') {
        $argList += 'Include_freethreaded=1'
    }

    Write-Host "Installing Python $FullVer to $TargetDir"
    $p = Start-Process -FilePath $installer -ArgumentList $argList -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "Python installer exited with $($p.ExitCode)"
    }

    if ($freethreaded -and $majMin -ge [version]'3.13') {
        $ft = Get-ChildItem -Path $TargetDir -Filter 'python*.*t.exe' -File -ErrorAction SilentlyContinue |
            Sort-Object Name |
            Select-Object -First 1
        if (-not $ft) {
            $ft = Get-ChildItem -Path $TargetDir -Filter '*t.exe' -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^python\d' } |
                Select-Object -First 1
        }
        if (-not $ft) {
            throw "Free-threaded python.exe not found under $TargetDir"
        }
        return $ft.FullName
    }

    $pyExe = Join-Path $TargetDir 'python.exe'
    if (-not (Test-Path $pyExe)) {
        throw "python.exe not found under $TargetDir"
    }
    return $pyExe
}

function Resolve-Bash {
    $cmd = Get-Command bash -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    foreach ($c in @('C:\Program Files\Git\bin\bash.exe', 'C:\Program Files\Git\usr\bin\bash.exe')) {
        if (Test-Path $c) {
            return $c
        }
    }
    throw 'bash not found (expected Git for Windows in the devcontainer)'
}

function Convert-ToUnixPath {
    param([string]$WinPath)
    if ($WinPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = ($Matches[2] -replace '\\', '/')
        return "/$drive/$rest"
    }
    return ($WinPath -replace '\\', '/')
}

function Test-Toolchain {
    foreach ($t in @('cmake', 'ninja', 'git', 'cl')) {
        if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
            throw "Required tool not found on PATH: $t"
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot
Write-Host "Repository root: $repoRoot"

Test-Toolchain

$spec = $PythonVersion.Trim()
$fullVer = Resolve-FullPythonVersion -Spec $spec
$targetDir = 'C:\python-ci'
if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
}
New-Item -ItemType Directory -Path $targetDir | Out-Null

$pythonExe = Install-PythonFromPythonOrg -Spec $spec -FullVer $fullVer -TargetDir $targetDir
Write-Host "Using Python: $pythonExe"
& $pythonExe --version

& $pythonExe -m pip install --upgrade pip

if ($Mode -eq 'modern') {
    $pkgs = @('pybind11', 'nanobind', 'numpy', 'ninja', 'cmake', 'awscli')
    & $pythonExe -m pip install @pkgs
    if ([version]($spec.TrimEnd('t').Trim()) -lt [version]'3.12') {
        & $pythonExe -m pip install 'typing-extensions'
    }
}
else {
    & $pythonExe -m pip install @('ninja', 'cmake', 'awscli')
}

$bash = Resolve-Bash
$repoRootUnix = Convert-ToUnixPath -WinPath $repoRoot
& git config --global --add safe.directory $repoRootUnix

$parallel = if ($env:PARALLEL) { $env:PARALLEL } else { '16' }
$pythonForBash = ($pythonExe -replace '\\', '/')
$modeArg = $Mode

$bashCmd = @"
set -euo pipefail
cd '$repoRootUnix'
chmod +x ci/*.sh || true
export PYTHON="$pythonForBash"
export PARALLEL="$parallel"
ci/build-windows.sh $modeArg
"@

Write-Host "Running build via bash (PARALLEL=$parallel, Mode=$modeArg)"
& $bash -lc $bashCmd
