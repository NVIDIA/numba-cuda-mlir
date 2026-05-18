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
