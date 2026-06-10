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

. (Join-Path $PSScriptRoot 'windows-common.ps1')

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

$env:PYTHON = $pythonForBash
$env:PARALLEL = $parallel

Write-Host "Running build via bash (PARALLEL=$parallel, Mode=$modeArg)"
& $bash -lc "cd '$repoRootUnix' && bash ci/build-windows-driver.sh '$modeArg'"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
