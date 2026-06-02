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
    foreach ($tool in @('cmake', 'ninja', 'git', 'cl')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Required tool not found on PATH: $tool"
        }
    }
}
