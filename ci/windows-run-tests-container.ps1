# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonVersion,
    [Parameter(Mandatory = $true)]
    [string]$CudaVersion,
    [Parameter(Mandatory = $true)]
    [ValidateSet('0', '1')]
    [string]$LocalCtk,
    [Parameter(Mandatory = $true)]
    [ValidateSet('run-tests', 'run-coverage-tests')]
    [string]$TestScript
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'windows-common.ps1')

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot
Write-Host "Repository root: $repoRoot"

& git config --global --add safe.directory (Convert-ToUnixPath -WinPath $repoRoot)

$targetDir = 'C:\python-ci'
if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
}
New-Item -ItemType Directory -Path $targetDir | Out-Null

$spec = $PythonVersion.Trim()
$pythonExe = Install-PythonViaNuGet -Spec $spec -TargetDir $targetDir
Write-Host "Using Python: $pythonExe"
& $pythonExe --version
& $pythonExe -m ensurepip --upgrade
& $pythonExe -m pip install --upgrade pip

$bash = Resolve-Bash
$repoRootUnix = Convert-ToUnixPath -WinPath $repoRoot
$pythonDir = Split-Path -Parent $pythonExe
$pythonScripts = Join-Path $pythonDir 'Scripts'
$pythonDirUnix = Convert-ToUnixPath -WinPath $pythonDir
$pythonScriptsUnix = Convert-ToUnixPath -WinPath $pythonScripts

$cudaMajor = ($CudaVersion -split '\.')[0]
$cudaMinor = ($CudaVersion -split '\.')[1]
$cudaPath = Join-Path $repoRoot 'cuda_toolkit'
$cudaPathUnix = Convert-ToUnixPath -WinPath $cudaPath

$env:PATH = "$pythonScripts;$pythonDir;$env:PATH"
if ($LocalCtk -eq '1') {
    if (-not (Test-Path (Join-Path $cudaPath 'bin'))) {
        throw "LOCAL_CTK=1 but mini CTK was not found at $cudaPath"
    }
    $env:CUDA_PATH = $cudaPath
    $env:CUDA_HOME = $cudaPath
    $env:PATH = "$(Join-Path $cudaPath 'bin');$env:PATH"
}

$bashCmd = @"
set -euo pipefail
cd '$repoRootUnix'
chmod +x ci/tools/* || true
export PATH="${pythonScriptsUnix}:${pythonDirUnix}:${repoRootUnix}/ci/tools:${repoRootUnix}/llvm-modern-install/lib:${repoRootUnix}/llvm-modern-install/bin:${repoRootUnix}/llvm7-install/lib:${repoRootUnix}/llvm7-install/bin:`$PATH"
export NUMBA_CUDA_MLIR_CUDA_ARTIFACTS_DIR='$repoRootUnix/dist'
export TEST_CUDA_MAJOR='$cudaMajor'
export TEST_CUDA_MINOR='$cudaMinor'
export LOCAL_CTK='$LocalCtk'
export SANITIZER_CMD=''
if [[ '$LocalCtk' == '1' ]]; then
  export CUDA_PATH='$cudaPath'
  export CUDA_HOME='$cudaPath'
  export PATH="$cudaPathUnix/bin:`$PATH"
else
  export NUMBA_CUDA_MLIR_TEST_WHEEL_ONLY=1
  export NUMBA_CUDA_MLIR_CUDA_TEST_WHEEL_ONLY=1
  export NUMBA_CUDA_TEST_WHEEL_ONLY=1
fi
'$TestScript'
"@

Write-Host "Running $TestScript inside Windows devcontainer"
& $bash -lc $bashCmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
