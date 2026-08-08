# Builds the plugin and the test harness.
#
# The system CMake (3.28) predates Visual Studio 18, and VS 18 Community is the
# only install here carrying a C++ toolset -- so we drive VS 18's own bundled
# CMake + Ninja from inside its developer environment instead.
param(
  [string]$Config = "Release",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$vsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"
$vcvars = "$vsRoot\VC\Auxiliary\Build\vcvars64.bat"
$cmakeDir = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake"
$cmake = "$cmakeDir\CMake\bin\cmake.exe"
$ninja = "$cmakeDir\Ninja\ninja.exe"

foreach ($p in @($vcvars, $cmake, $ninja)) {
  if (-not (Test-Path $p)) { throw "Not found: $p" }
}

$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

# vcvars64 only exports into a cmd session, so configure and build must both
# run inside that same session.
$script = @"
@echo off
rem vcvars64 shells out to vswhere, which is not on PATH by default.
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "$vcvars" >nul || exit /b 1
"$cmake" -S "$root" -B "$build" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="$ninja" ^
  -DCMAKE_BUILD_TYPE=$Config || exit /b 1
"$cmake" --build "$build" || exit /b 1
"@

$bat = Join-Path $env:TEMP "rta_build.bat"
Set-Content -Path $bat -Value $script -Encoding ASCII
& cmd.exe /c $bat
$code = $LASTEXITCODE
Remove-Item $bat -ErrorAction SilentlyContinue
if ($code -ne 0) { throw "Build failed with exit code $code" }

Write-Host ""
Write-Host "Built:" -ForegroundColor Green
Write-Host "  $build\bundle\TextAnimator.ofx.bundle"
Write-Host "  $build\segtest.exe"
