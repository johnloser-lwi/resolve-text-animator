# Install TextAnimator.ofx.bundle into the system OFX plugin directory.
#
# Self-elevating: if run from a normal shell it relaunches itself via UAC.
# You can also just double-click Install-TextAnimator.cmd next to this file.
#
# Resolve scans for OFX plugins only at launch, so it must be restarted after
# installing, and it holds a file lock on any plugin build it has already loaded.
[CmdletBinding()]
param(
    [string]$Destination = 'C:\Program Files\Common Files\OFX\Plugins',

    # Build before installing.
    [switch]$Build,

    # Close Resolve automatically instead of aborting.
    [switch]$Force,

    # Set internally when relaunching elevated; keeps the new window open.
    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'
$repo   = $PSScriptRoot
$bundle = Join-Path $repo 'build\bundle\TextAnimator.ofx.bundle'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok  ($msg) { Write-Host "    $msg" -ForegroundColor Green }

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# Probe rather than assume: an OFX directory with relaxed ACLs needs no
# elevation, and prompting for rights we do not need is just friction.
function Test-Writable([string]$dir) {
    try {
        if (-not (Test-Path $dir)) {
            $parent = Split-Path -Parent $dir
            if (-not (Test-Path $parent)) { return $false }
            New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
        }
        $probe = Join-Path $dir ".rta-write-probe-$PID"
        New-Item -ItemType File -Path $probe -ErrorAction Stop | Out-Null
        Remove-Item $probe -Force -ErrorAction SilentlyContinue
        return $true
    } catch { return $false }
}

$needsElevation = -not (Test-Writable $Destination)

# --- Build BEFORE elevating --------------------------------------------------
# Deliberately unelevated: building as administrator leaves build\ owned by the
# admin account, and every later non-elevated build then fails to overwrite it.
if ($Build -and -not $isAdmin) {
    Write-Step "Building..."
    & (Join-Path $repo 'build.ps1')
}

# --- Elevate if needed -------------------------------------------------------
if ($needsElevation -and -not $isAdmin) {
    Write-Step "Administrator rights required for $Destination - requesting elevation..."

    # -Build is intentionally NOT forwarded; the build already happened above.
    $argList = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-Destination', "`"$Destination`"",
        '-Elevated'
    )
    if ($Force) { $argList += '-Force' }

    try {
        $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $argList `
                              -Verb RunAs -PassThru -Wait
    } catch {
        Write-Host ""
        Write-Host "Elevation was declined or failed. Nothing was installed." -ForegroundColor Yellow
        exit 1
    }

    exit $proc.ExitCode
}

try {
    # --- Preconditions -------------------------------------------------------
    Write-Step "Checking bundle..."
    if (-not (Test-Path $bundle)) {
        throw "Bundle not found at:`n  $bundle`nRun .\build.ps1 first, or re-run this with -Build."
    }

    $payload = Join-Path $bundle 'Contents\Win64\TextAnimator.ofx'
    if (-not (Test-Path $payload)) {
        throw "Bundle is malformed - missing Contents\Win64\TextAnimator.ofx"
    }
    Write-Ok "Found $([math]::Round((Get-Item $payload).Length / 1KB)) KB plugin binary."

    # --- Resolve -------------------------------------------------------------
    # Only an actual file lock matters. Resolve locks a plugin DLL it has loaded,
    # but a first-time install of a plugin Resolve has never seen copies fine
    # while it runs. So warn, try, and only insist on closing if the copy fails.
    $resolve = Get-Process -Name 'Resolve' -ErrorAction SilentlyContinue
    if ($resolve -and $Force) {
        Write-Step "Closing DaVinci Resolve..."
        $resolve.CloseMainWindow() | Out-Null
        if (-not $resolve.WaitForExit(30000)) {
            throw "Resolve did not close within 30s. Close it manually and re-run."
        }
        Write-Ok "Closed."
        $resolve = $null
    }
    elseif ($resolve) {
        Write-Warning "DaVinci Resolve is running. Attempting the install anyway; it will fail only if Resolve has already loaded a previous build of this plugin."
    }

    # --- Install -------------------------------------------------------------
    Write-Step "Installing to $Destination"
    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    }

    $target = Join-Path $Destination 'TextAnimator.ofx.bundle'
    try {
        if (Test-Path $target) {
            Write-Ok "Replacing existing installation."
            Remove-Item -Recurse -Force $target -ErrorAction Stop
        }
        Copy-Item $bundle -Destination $Destination -Recurse -Force -ErrorAction Stop
    }
    catch {
        if ($resolve) {
            throw "Could not write to $target because DaVinci Resolve has the plugin locked.`nClose Resolve and re-run, or pass -Force to have this script close it for you.`n`nUnderlying error: $($_.Exception.Message)"
        }
        throw
    }

    # --- Verify --------------------------------------------------------------
    $installed = Join-Path $target 'Contents\Win64\TextAnimator.ofx'
    if (-not (Test-Path $installed)) { throw "Copy appeared to succeed but $installed is missing." }

    $srcHash = (Get-FileHash $payload   -Algorithm SHA256).Hash
    $dstHash = (Get-FileHash $installed -Algorithm SHA256).Hash
    if ($srcHash -ne $dstHash) { throw "Installed binary does not match the built binary." }

    Write-Ok "Verified: $installed"
    Write-Host ""
    Write-Host "Installed successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next:"
    Write-Host "  1. Start DaVinci Resolve."
    Write-Host "  2. Put a Text+ on its own track over any clip."
    Write-Host "  3. Apply OpenFX > Filters > Text > 'Text Animator' to the Text+ clip."
    Write-Host "  4. Turn on 'Show Detection' first to confirm the words are found."

    $exitCode = 0
}
catch {
    Write-Host ""
    Write-Host "INSTALL FAILED" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    $exitCode = 1
}

if ($Elevated) {
    Write-Host ""
    Read-Host "Press Enter to close"
}
exit $exitCode
