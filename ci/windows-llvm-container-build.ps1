# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonVersion,
    [Parameter(Mandatory = $true)]
    [ValidateSet('modern', 'llvm7')]
    [string]$Mode,
    [Parameter(Mandatory = $true)]
    [string]$UvPath
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

$pythonExe = Install-HardenedPython -Spec $spec -TargetDir $targetDir -UvPath $UvPath
Write-Host "Using Python: $pythonExe"
& $pythonExe --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Mode -eq 'modern') {
    # MLIR asks for nanobind 2.9 and nanobind's CMake config only accepts a
    # matching major version, so 3.x fails the configure step outright.
    $pkgs = @('pybind11', 'nanobind~=2.9', 'numpy', 'ninja', 'cmake', 'awscli')
    & $pythonExe -m pip install @pkgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ([version]($spec.TrimEnd('t').Trim()) -lt [version]'3.12') {
        & $pythonExe -m pip install 'typing-extensions'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
else {
    & $pythonExe -m pip install @('ninja', 'cmake', 'awscli')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$bash = Resolve-Bash
$repoRootUnix = Convert-ToUnixPath -WinPath $repoRoot
& git config --global --add safe.directory $repoRootUnix
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

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
