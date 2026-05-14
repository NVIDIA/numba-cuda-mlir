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

function Install-PythonViaNuGet {
    param(
        [string]$Spec,
        [string]$TargetDir
    )
    $freethreaded = $Spec -match 't$'
    $baseVersion = $Spec.TrimEnd('t').Trim()

    $nugetExe = Join-Path $env:TEMP 'nuget.exe'
    if (-not (Test-Path $nugetExe)) {
        Write-Host 'Downloading nuget.exe'
        Invoke-WebRequest -Uri 'https://dist.nuget.org/win-x86-commandline/latest/nuget.exe' -OutFile $nugetExe -UseBasicParsing
    }

    if ($freethreaded) {
        $packageId = 'python-freethreaded'
    }
    else {
        $packageId = 'python'
    }

    Write-Host "Installing $packageId $baseVersion via NuGet to $TargetDir"
    $nugetArgs = @(
        'install', $packageId,
        '-Version', $baseVersion,
        '-OutputDirectory', $TargetDir,
        '-ExcludeVersion'
    )
    $p = Start-Process -FilePath $nugetExe -ArgumentList $nugetArgs -Wait -NoNewWindow -PassThru
    if ($p.ExitCode -ne 0) {
        Write-Host "Exact version $baseVersion not found, trying version prefix"
        $nugetArgs = @(
            'install', $packageId,
            '-Version', "[${baseVersion},${baseVersion}.99999]",
            '-OutputDirectory', $TargetDir,
            '-ExcludeVersion'
        )
        $p = Start-Process -FilePath $nugetExe -ArgumentList $nugetArgs -Wait -NoNewWindow -PassThru
        if ($p.ExitCode -ne 0) {
            throw "Failed to install $packageId $baseVersion via NuGet"
        }
    }

    $pkgDir = Join-Path $TargetDir $packageId
    $toolsDir = Join-Path $pkgDir 'tools'
    if (-not (Test-Path $toolsDir)) {
        throw "NuGet package installed but tools/ directory not found under $pkgDir"
    }

    $pyExe = Join-Path $toolsDir 'python.exe'
    if (-not (Test-Path $pyExe)) {
        throw "python.exe not found under $toolsDir"
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
$targetDir = 'C:\python-ci'
if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
}
New-Item -ItemType Directory -Path $targetDir | Out-Null

$pythonExe = Install-PythonViaNuGet -Spec $spec -TargetDir $targetDir
Write-Host "Using Python: $pythonExe"
& $pythonExe --version

& $pythonExe -m ensurepip --upgrade
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
