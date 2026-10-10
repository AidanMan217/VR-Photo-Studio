; VR Photo Studio - Windows installer (Inno Setup 6.x; also compiles with 7.x)
;
; Build the PyInstaller bundle first (pyinstaller packaging\vrphoto.spec), then:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=0.1.0 packaging\installer.iss
; Optional defines:
;   /DAppVersionNumeric=0.1.0.0   numeric version for the setup's version resource
;   /DBundleDir=<path>            the onedir bundle (default ..\dist\VRPhotoStudio)
;   /DOutputDir=<path>            where the setup .exe goes (default packaging\Output)
;
; Output: VRPhotoStudio-Setup-<AppVersion>.exe
; Default: per-user install into %LOCALAPPDATA%\Programs\VR Photo Studio (no admin rights
; needed); the user may choose "install for all users" in the first dialog.
; Silent per-user install (as used by CI):
;   VRPhotoStudio-Setup-0.1.0.exe /VERYSILENT /CURRENTUSER /SUPPRESSMSGBOXES /NORESTART /DIR="C:\VRPS"

; 6.3+ is needed for ArchitecturesAllowed=x64compatible and for UTF-8 licence files without a BOM.
#if VER < EncodeVer(6, 3, 0)
  #error Inno Setup 6.3 or newer is required - download it from https://jrsoftware.org/isdl.php
#endif

#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif
#ifndef BundleDir
  #define BundleDir "..\dist\VRPhotoStudio"
#endif
#ifndef OutputDir
  #define OutputDir "Output"
#endif

#define AppName "VR Photo Studio"
#define AppExe "VRPhotoStudio.exe"
#define CliExe "vrphoto-cli.exe"
#define AppPublisher "AidanMan217"
#define AppURL "https://github.com/AidanMan217/VR-Photo-Studio"
; Must match vrphoto.core.config.data_root() on Windows: %LOCALAPPDATA%\VRPhotoStudio
#define DataDirName "VRPhotoStudio"
; Inbound ports used by the WebXR viewer server (HTTPS for the Quest, HTTP for localhost/USB)
#define ViewerPorts "8443,8080"
; Same name as vrphoto.webxr.net.FIREWALL_RULE_NAME, so the uninstaller also removes a rule the app added
#define FirewallRuleName "VR Photo Studio"

; Fail early with a clear message if the PyInstaller bundle is missing.
#if Pos(":", BundleDir) > 0
  #define BundleCheckDir BundleDir
#else
  #define BundleCheckDir AddBackslash(SourcePath) + BundleDir
#endif
#if FileExists(BundleCheckDir + "\" + AppExe) == 0
  #error App bundle not found (VRPhotoStudio.exe). Run "pyinstaller packaging\vrphoto.spec" first, or pass /DBundleDir=<folder>.
#endif
#if FileExists(BundleCheckDir + "\LICENSE.txt") == 0
  #error LICENSE.txt missing from the app bundle - it is written by packaging\vrphoto.spec.
#endif

[Setup]
; AppId identifies the app for upgrades/uninstall - never change it.
AppId={{6F3C2B1E-8A47-4D5B-9E21-7C0D4A9B3F58}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
AppCopyright=Copyright (c) 2026 {#AppPublisher}
#ifdef AppVersionNumeric
VersionInfoVersion={#AppVersionNumeric}
#endif
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
VersionInfoCompany={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user by default; the "for all users / just me" dialog lets the user elevate.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsedUserAreasWarning=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=VRPhotoStudio-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
LicenseFile={#BundleDir}\LICENSE.txt
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
ShowLanguageDialog=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
english.NetworkGroup=Quest headset connection:
english.FirewallTask=Allow Quest headset to connect (Windows Firewall, private networks only)
english.FirewallStatus=Allowing the Quest headset through Windows Firewall...
english.AppComment=Turn photos into 3D, VR180, 360 and walk-around scenes for Meta Quest
english.AskDeleteData=Do you also want to delete VR Photo Studio's downloaded AI models, add-ons, settings and logs?%n%n%1%n%nThese can take up many gigabytes. Choose Yes to free the space, or No to keep them (useful if you plan to reinstall).%n%nYour photos and the VR files you created are NOT affected either way.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
; Adding a firewall rule needs administrator rights, so this is only offered when Setup
; runs elevated ("install for all users").  Otherwise the app offers it on first use.
Name: "firewall"; Description: "{cm:FirewallTask}"; GroupDescription: "{cm:NetworkGroup}"; Check: IsAdmin

[InstallDelete]
; Remove the previous version's runtime so no stale DLLs survive an upgrade.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; Comment: "{cm:AppComment}"; AppUserModelID: "VRPhotoStudio.App"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Comment: "{cm:AppComment}"; AppUserModelID: "VRPhotoStudio.App"; Tasks: desktopicon

[Run]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""{#FirewallRuleName}"""; Flags: runhidden waituntilterminated; Tasks: firewall
; Windows creates per-program *block* rules when someone dismisses its "allow access" prompt, and
; block rules win over allow rules - remove any left over from an earlier install (as the app does).
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=all dir=in program=""{app}\{#AppExe}"""; Flags: runhidden waituntilterminated; Tasks: firewall
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""{#FirewallRuleName}"" dir=in action=allow protocol=TCP localport={#ViewerPorts} profile=private,domain description=""Lets your Meta Quest open the VR Photo Studio viewer over Wi-Fi"""; StatusMsg: "{cm:FirewallStatus}"; Flags: runhidden waituntilterminated; Tasks: firewall
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
Type: dirifempty; Name: "{app}"

[Code]
const
  DataDirName = '{#DataDirName}';
  FirewallRuleName = '{#FirewallRuleName}';

function DataDir(): String;
begin
  Result := ExpandConstant('{localappdata}\') + DataDirName;
end;

procedure RemoveFirewallRule();
var
  ResultCode: Integer;
begin
  { The rule may have been added by Setup or later by the app; deleting a missing rule is harmless.
    Also remove the per-program rules Windows created for our executables (its "allow access" prompt). }
  if IsAdmin() then
  begin
    Exec(ExpandConstant('{sys}\netsh.exe'),
      'advfirewall firewall delete rule name="' + FirewallRuleName + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec(ExpandConstant('{sys}\netsh.exe'),
      'advfirewall firewall delete rule name=all program="' + ExpandConstant('{app}\{#AppExe}') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec(ExpandConstant('{sys}\netsh.exe'),
      'advfirewall firewall delete rule name=all program="' + ExpandConstant('{app}\{#CliExe}') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Dir: String;
begin
  case CurUninstallStep of
    usUninstall:
      RemoveFirewallRule();
    usPostUninstall:
      begin
        Dir := DataDir();
        if DirExists(Dir) then
        begin
          { Default answer (also for /SUPPRESSMSGBOXES silent uninstalls) is No: keep the data. }
          if SuppressibleMsgBox(FmtMessage(CustomMessage('AskDeleteData'), [Dir]),
               mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
          begin
            if not DelTree(Dir, True, True, True) then
              Log('Could not completely delete ' + Dir);
          end;
        end;
      end;
  end;
end;
