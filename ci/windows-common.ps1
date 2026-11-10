# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function Import-EnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Get-Content $Path | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim()
        }
    }
}

function Install-HardenedPython {
    param(
        [string]$Spec,
        [string]$TargetDir,
        [string]$UvPath
    )

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    # The caller passes setup-uv's verified binary through a read-only mount.
    $uvExe = (Resolve-Path -LiteralPath $UvPath).Path
    if (-not (Test-Path -LiteralPath $uvExe -PathType Leaf)) {
        throw "Verified uv executable not found: $UvPath"
    }
    $TargetDir = [System.IO.Path]::GetFullPath($TargetDir)

    $oldNoConfig = $env:UV_NO_CONFIG
    $oldInstallDir = $env:UV_PYTHON_INSTALL_DIR
    $oldPython = $env:PYTHON
    try {
        $env:UV_NO_CONFIG = '1'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $TargetDir 'managed-python'
        & $uvExe python install --no-bin --no-registry $Spec | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "uv python install failed with exit code $LASTEXITCODE" }
        $managedPython = & $uvExe python find --managed-python --no-project $Spec
        if ($LASTEXITCODE -ne 0) { throw "uv python find failed with exit code $LASTEXITCODE" }
        if (-not ($managedPython -is [string]) -or -not (Test-Path -LiteralPath $managedPython -PathType Leaf)) {
            throw 'uv did not return a managed Python executable'
        }
        $venvDir = Join-Path $TargetDir 'venv'
        & $managedPython -m venv $venvDir | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed with exit code $LASTEXITCODE" }
        $pythonExe = Join-Path $venvDir 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
            throw "Venv Python executable not found: $pythonExe"
        }
        $env:PYTHON = $pythonExe
        & $pythonExe (Join-Path $repoRoot 'ci/tools/check_python_security.py') | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Python security check failed with exit code $LASTEXITCODE" }
        return $pythonExe
    }
    finally {
        $env:UV_NO_CONFIG = $oldNoConfig
        $env:UV_PYTHON_INSTALL_DIR = $oldInstallDir
        $env:PYTHON = $oldPython
    }
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
    foreach ($tool in @('cmake', 'ninja', 'git', 'cl')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required tool not found on PATH: $tool"
        }
    }
}
