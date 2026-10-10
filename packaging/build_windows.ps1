<#
.SYNOPSIS
    Build VR Photo Studio for Windows on your own PC.

.DESCRIPTION
    1. creates a build virtual environment (.venv-build) with Python 3.12
    2. installs the app and its build tools (pip install .[dev])
    3. draws the icon (packaging\make_icon.py)
    4. runs PyInstaller (packaging\vrphoto.spec) -> dist\VRPhotoStudio\
    5. checks the bundle and runs "vrphoto-cli.exe --selftest --json"
    6. compiles the installer with Inno Setup (if installed) -> packaging\Output\VRPhotoStudio-Setup-<ver>.exe
    7. optionally zips the app folder (-Zip)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

.EXAMPLE
    .\packaging\build_windows.ps1 -Clean -Zip

.EXAMPLE
    .\packaging\build_windows.ps1 -Python "C:\Python312\python.exe" -SkipInstaller
#>
[CmdletBinding()]
param(
    # Python used to create the build venv. Default: "py -3.12" (Python launcher), else "python".
    [string]$Python = "",
    # Build virtual environment folder (relative to the repository root).
    [string]$VenvDir = ".venv-build",
    # Delete previous build output (build\, dist\) and recreate the venv first.
    [switch]$Clean,
    # Do not run the frozen self-test.
    [switch]$SkipSelftest,
    # Do not build the Inno Setup installer.
    [switch]$SkipInstaller,
    # Also create packaging\Output\VRPhotoStudio-<ver>-win64.zip
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Stop-Build([string]$Message) {
    Write-Host ""
    Write-Host "BUILD FAILED: $Message" -ForegroundColor Red
    exit 1
}

function Invoke-Native {
    # Runs a native command and stops the build if it fails (PowerShell 5 does not do that by itself).
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$ArgList = @())
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        Stop-Build "'$Exe $($ArgList -join ' ')' exited with code $LASTEXITCODE"
    }
}

function Find-Iscc {
    $cmd = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @()
    foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles, (Join-Path $env:LOCALAPPDATA "Programs"))) {
        if ($base) {
            foreach ($ver in @("6", "7")) {
                $candidates += (Join-Path $base "Inno Setup $ver\ISCC.exe")
            }
        }
    }
    foreach ($key in @(
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
            "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
            "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 7_is1",
            "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 7_is1",
            "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 7_is1")) {
        $item = Get-ItemProperty -Path $key -ErrorAction SilentlyContinue
        if ($item -and ($item.PSObject.Properties.Name -contains "InstallLocation")) {
            $candidates += (Join-Path $item.InstallLocation "ISCC.exe")
        }
    }
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return $null
}

# ------------------------------------------------------------------ 1. Python + venv
Write-Step "Preparing the build environment ($VenvDir)"
if ($Clean -and (Test-Path $VenvDir)) {
    Remove-Item -Recurse -Force $VenvDir
}
$VenvPython = Join-Path $Root (Join-Path $VenvDir "Scripts\python.exe")
if (-not (Test-Path $VenvPython)) {
    $pyExe = $null
    $pyArgs = @()
    if ($Python) {
        $parts = $Python.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
        $pyExe = $parts[0]
        if ($parts.Length -gt 1) { $pyArgs = @($parts[1..($parts.Length - 1)]) }
    }
    elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $pyExe = "py"; $pyArgs = @("-3.12")
    }
    elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
        $pyExe = "python"
    }
    else {
        Stop-Build "Python 3.12 was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') and run this script again."
    }
    Invoke-Native $pyExe ($pyArgs + @("-m", "venv", $VenvDir))
}
Invoke-Native $VenvPython @("-c", "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'; print(sys.version)")

# ------------------------------------------------------------------ 2. dependencies
Write-Step "Installing VR Photo Studio and build tools (this can take a few minutes the first time)"
Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Native $VenvPython @("-m", "pip", "install", ".[dev]")

$Version = & $VenvPython "packaging\spec_helpers.py" "version"
if ($LASTEXITCODE -ne 0 -or -not $Version) {
    Stop-Build "could not determine the version (see the message above). src\vrphoto\__init__.py and pyproject.toml must contain the same version."
}
$Version = "$Version".Trim()
$NumericVersion = "$(& $VenvPython "packaging\spec_helpers.py" "numeric-version")".Trim()
Write-Host "Version: $Version ($NumericVersion)"

# ------------------------------------------------------------------ 3. icon
Write-Step "Drawing the icon"
Invoke-Native $VenvPython @("packaging\make_icon.py")

# ------------------------------------------------------------------ 4. PyInstaller
Write-Step "Bundling the app with PyInstaller"
if ($Clean) {
    foreach ($d in @("build\pyinstaller", "dist\VRPhotoStudio")) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    }
}
$env:VRPHOTO_STRICT_BUNDLE = "1"
Invoke-Native $VenvPython @("-m", "PyInstaller", "packaging\vrphoto.spec", "--noconfirm",
    "--distpath", "dist", "--workpath", "build\pyinstaller")
Remove-Item Env:\VRPHOTO_STRICT_BUNDLE -ErrorAction SilentlyContinue

$Bundle = Join-Path $Root "dist\VRPhotoStudio"
Invoke-Native $VenvPython @("packaging\spec_helpers.py", "check", $Bundle)

# ------------------------------------------------------------------ 5. self-test
if (-not $SkipSelftest) {
    Write-Step "Running the self-test of the bundled app"
    Invoke-Native (Join-Path $Bundle "vrphoto-cli.exe") @("--selftest", "--json")
}

# ------------------------------------------------------------------ 6. installer
$OutDir = Join-Path $Root "packaging\Output"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (-not $SkipInstaller) {
    Write-Step "Building the installer with Inno Setup"
    $Iscc = Find-Iscc
    if (-not $Iscc) {
        Write-Warning ("Inno Setup 6 was not found, so no installer was built. Install it from " +
            "https://jrsoftware.org/isdl.php (or: winget install JRSoftware.InnoSetup) and run this script again. " +
            "The app itself is ready in $Bundle")
    }
    else {
        Invoke-Native $Iscc @("/Q", "/DAppVersion=$Version", "/DAppVersionNumeric=$NumericVersion",
            "/DBundleDir=$Bundle", "/DOutputDir=$OutDir", "packaging\installer.iss")
        Write-Host "Installer: $(Join-Path $OutDir "VRPhotoStudio-Setup-$Version.exe")" -ForegroundColor Green
    }
}

# ------------------------------------------------------------------ 7. zip
if ($Zip) {
    Write-Step "Zipping the app folder"
    $ZipPath = Join-Path $OutDir "VRPhotoStudio-$Version-win64.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Compress-Archive -Path $Bundle -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "Zip: $ZipPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. App folder: $Bundle" -ForegroundColor Green
