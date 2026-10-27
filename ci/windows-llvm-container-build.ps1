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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Bootstrap pip 26.2.2 from a hash-pinned URL before any other use of pip:
# older pip releases still trust the compromised eMudhra/emSign intermediate
# CAs via their vendored certifi and cannot upgrade themselves safely. Keep
# the URL in sync with .github/actions/harden-pip/action.yml.
# See https://discuss.python.org/t/109576.
# $ErrorActionPreference = 'Stop' does not trap native-command exit codes
# in Windows PowerShell 5.1, so these two installs need an explicit check:
# a silent failure on either would let the hardening be bypassed by a
# subsequent command that resets $LASTEXITCODE.
& $pythonExe -m pip install --disable-pip-version-check "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Mode -eq 'modern') {
    # MLIR asks for nanobind 2.9 and nanobind's CMake config only accepts a
    # matching major version, so 3.x fails the configure step outright.
    $pkgs = @('pybind11', 'nanobind~=2.9', 'numpy', 'ninja', 'cmake', 'awscli')
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
