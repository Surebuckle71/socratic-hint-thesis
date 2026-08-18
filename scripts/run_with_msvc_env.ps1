# Sets up the MSVC + Windows SDK environment that Triton (used by Unsloth) needs to
# JIT-compile its CUDA driver stub on native Windows, then runs the given command.
#
# vcvarsall.bat/vcvars64.bat itself fails to parse on this machine (a stray "NVIDIA"
# token in an already-set environment variable trips up its batch parsing), so the
# needed variables are set directly here instead of delegating to that script.
#
# Usage: powershell -File run_with_msvc_env.ps1 -Command "pytest tests/training -v -m gpu"
#        powershell -File run_with_msvc_env.ps1 -VenvPath .venv313 -Command "..."

param(
    [Parameter(Mandatory = $true)]
    [string]$Command,

    # Project-local venv to activate before running $Command, if any. Leave blank to
    # use whatever Python is already on PATH. Defaults to this repo's .venv313
    # (absolute path, since $PSScriptRoot is unreliable when this script is invoked
    # via `powershell -File` from some nested-shell contexts).
    [string]$VenvPath = "C:\Users\rampr\Documents\GitHub\socratic-hint-thesis\.venv313"
)

$ErrorActionPreference = "Stop"

if ($VenvPath -and (Test-Path "$VenvPath\Scripts\Activate.ps1")) {
    # Dot-source (not &) so the venv's PATH/env changes persist in this scope.
    . "$VenvPath\Scripts\Activate.ps1"
}

$msvcRoot = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$sdk = "C:\Program Files (x86)\Windows Kits\10"
$sdkVer = Get-ChildItem "$sdk\Include" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name

# Resolve triton's bundled cuda.lib directory dynamically from whichever python/venv
# is now active, rather than hardcoding a path tied to one interpreter.
$tritonNvidiaLib = (python -c "import os, triton; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'lib', 'x64'))").Trim()

$env:PATH = "$msvcRoot\bin\Hostx64\x64;$sdk\bin\$sdkVer\x64;" + $env:PATH
$env:INCLUDE = "$msvcRoot\include;$sdk\Include\$sdkVer\ucrt;$sdk\Include\$sdkVer\shared;$sdk\Include\$sdkVer\um;$sdk\Include\$sdkVer\winrt"
$env:LIB = "$tritonNvidiaLib;$msvcRoot\lib\x64;$sdk\Lib\$sdkVer\ucrt\x64;$sdk\Lib\$sdkVer\um\x64"
$env:CC = "$msvcRoot\bin\Hostx64\x64\cl.exe"

Invoke-Expression $Command
exit $LASTEXITCODE
