; ============================================================
;  ParallelFinder — Inno Setup Script
;  Author  : Pozit1vchic
;  Version : v12.6  (numeric: 12.6.0.1)
;  Style   : modern, bilingual (RU / EN)
; ============================================================
;
;  WIZARD IMAGES (optional, remove if files are absent):
;    WizardImageFile      — 164x314 px  BMP  (left banner)
;    WizardSmallImageFile — 55x58  px  BMP  (top-right logo)
;  Place them next to this .iss or adjust paths below.
;  If absent — comment out or delete those two lines.
;
; ============================================================

#define AppName        "ParallelFinder"
#define AppVer         "v12.6"
#define AppVerNumeric  "12.6.0.1"
#define AppPublisher   "Pozit1vchic"
#define AppExeName     "run_app.vbs"
#define AppMainScript  "main.py"

[Setup]
AppName={#AppName}
AppVersion={#AppVer}
AppPublisher={#AppPublisher}
AppContact=https://github.com/Pozit1vchic
AppCopyright=© 2024-2025 {#AppPublisher}

VersionInfoVersion={#AppVerNumeric}
VersionInfoProductVersion={#AppVerNumeric}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}

DefaultDirName={autopf}\{#AppName}
DisableDirPage=no
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputBaseFilename=ParallelFinder_Setup_v12.6
Compression=lzma2/ultra64
SolidCompression=yes

SetupIconFile=icons\icon.ico
UninstallDisplayIcon={app}\icons\icon.ico

WizardStyle=modern
WizardResizable=no

; ── Wizard images ─────────────────────────────────────────────
; Uncomment once you have the BMP files prepared:
; WizardImageFile=wizard_banner.bmp       ; 164×314 px, 24-bit BMP
; WizardSmallImageFile=wizard_logo.bmp    ; 55×58 px, 24-bit BMP
; ─────────────────────────────────────────────────────────────

PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=commandline
SetupLogging=yes
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
CreateUninstallRegKey=yes
DiskSpanning=no

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]

; ---- Russian ------------------------------------------------
ru.PageTorchTitle=Вычислительный бэкенд — Torch
ru.PageTorchSubtitle=Выберите, как ParallelFinder будет использовать нейросети
ru.PageFaissTitle=Вычислительный бэкенд — FAISS
ru.PageFaissSubtitle=Выберите режим работы поискового индекса FAISS
ru.PageLibsTitle=Папка для Python-зависимостей
ru.PageLibsSubtitle=Библиотеки будут установлены сюда, не затрагивая систему
ru.PageTempTitle=Временная папка установки
ru.PageTempSubtitle=Используется pip-ом для кэша и распаковки пакетов

ru.TorchOpt0=CPU  (~1.2 ГБ)  —  нет NVIDIA GPU или не уверены в выборе
ru.TorchOpt1=GPU  (CUDA 11.8, ~3.0 ГБ)  —  старые или слабые видеокарты
ru.TorchOpt2=GPU  (CUDA 12.8, ~3.5 ГБ)  —  новые системы и актуальные драйверы

ru.FaissOpt0=CPU  (стабильно, рекомендуется)
ru.FaissOpt1=GPU  (экспериментально — на Windows может не работать)
ru.FaissGpuWarning=Внимание: FAISS-GPU для Windows пока нестабилен. Будет установлен faiss-cpu. Вы можете вручную собрать FAISS-GPU после завершения установки.

ru.LibsPathLabel=Папка для Python-библиотек:
ru.LibsPathDesc=Укажите папку, куда будут установлены Python-зависимости приложения. Путь должен содержать только латинские символы. Оставьте пустым, чтобы использовать папку по умолчанию:
ru.LibsPathDefault=(по умолчанию: {app}\pythonlibs)

ru.TempPathLabel=Временная папка (для pip):
ru.TempPathDesc=Укажите папку для временных файлов pip. Путь должен содержать только латинские символы. Оставьте пустым, чтобы использовать системный TEMP.
ru.TempPathDefault=(по умолчанию: системный TEMP)

ru.CyrillicWarningLibs=Папка зависимостей содержит символы кириллицы или пробелы. Это может привести к сбоям при установке Python-пакетов. Пожалуйста, выберите путь, содержащий только латинские буквы, цифры, дефисы и подчёркивания (например: C:\ParallelFinderLibs).
ru.CyrillicWarningTemp=Временная папка содержит кириллицу или пробелы, что может помешать работе pip. Рекомендуется использовать путь из латинских символов (например: C:\Temp\pftemp) или оставить поле пустым.

ru.PythonNotFound=Python не обнаружен в системе. Для работы {#AppName} требуется Python 3.10. Нажмите «Да», чтобы установить поставляемый Python 3.10, или «Нет», чтобы установить его самостоятельно (https://www.python.org/).
ru.PythonFound=Python найден: 
ru.PythonFoundMsg=Python уже установлен в системе. Повторная установка не требуется.

ru.TaskDesktop=Создать ярлык на Рабочем столе
ru.TaskRunAfter=Запустить {#AppName} после завершения установки

ru.StatusInstallPython=Установка Python 3.10...
ru.StatusInstallDeps=Установка Python-зависимостей (это может занять 5–20 минут)...
ru.StatusWriteConfig=Сохранение конфигурации...

ru.ReadyTorchLabel=Torch-бэкенд:
ru.ReadyFaissLabel=FAISS-бэкенд:
ru.ReadyLibsLabel=Папка зависимостей:
ru.ReadyTempLabel=Временная папка pip:
ru.ReadyTempSystem=(системный TEMP)

ru.UninstOldLibs=Обнаружена предыдущая папка зависимостей. Она будет удалена перед новой установкой.

; ---- English ------------------------------------------------
en.PageTorchTitle=Compute Backend — Torch
en.PageTorchSubtitle=Choose how ParallelFinder will run neural network inference
en.PageFaissTitle=Search Backend — FAISS
en.PageFaissSubtitle=Choose how the FAISS similarity index will be computed
en.PageLibsTitle=Python Dependencies Folder
en.PageLibsSubtitle=Libraries will be installed here, keeping your system Python clean
en.PageTempTitle=Temporary Installation Folder
en.PageTempSubtitle=Used by pip for package cache and extraction

en.TorchOpt0=CPU  (~1.2 GB)  —  no NVIDIA GPU, or not sure which to pick
en.TorchOpt1=GPU  (CUDA 11.8, ~3.0 GB)  —  older or lower-end GPUs
en.TorchOpt2=GPU  (CUDA 12.8, ~3.5 GB)  —  modern systems with up-to-date drivers

en.FaissOpt0=CPU  (stable, recommended)
en.FaissOpt1=GPU  (experimental — may not work on Windows)
en.FaissGpuWarning=Note: FAISS-GPU for Windows is currently unstable. faiss-cpu will be installed instead. You can build FAISS-GPU manually after installation if needed.

en.LibsPathLabel=Python libraries folder:
en.LibsPathDesc=Choose where Python dependencies will be installed. The path must contain only ASCII characters (no spaces, no Cyrillic). Leave blank to use the default location:
en.LibsPathDefault=(default: {app}\pythonlibs)

en.TempPathLabel=Temporary folder (for pip):
en.TempPathDesc=Specify a folder for pip temporary files. ASCII characters only. Leave blank to use the system TEMP.
en.TempPathDefault=(default: system TEMP)

en.CyrillicWarningLibs=The dependencies folder path contains Cyrillic characters or spaces, which can break Python package installation. Please choose a path with only ASCII letters, digits, hyphens and underscores (e.g. C:\ParallelFinderLibs).
en.CyrillicWarningTemp=The temporary folder path contains Cyrillic characters or spaces, which may cause pip to fail. Use an ASCII-only path (e.g. C:\Temp\pftemp) or leave the field blank.

en.PythonNotFound=Python was not found on this system. {#AppName} requires Python 3.10. Click Yes to install the bundled Python 3.10, or No to install it yourself (https://www.python.org/).
en.PythonFound=Python detected: 
en.PythonFoundMsg=Python is already installed. No additional Python installation needed.

en.TaskDesktop=Create a Desktop shortcut
en.TaskRunAfter=Launch {#AppName} after setup completes

en.StatusInstallPython=Installing Python 3.10...
en.StatusInstallDeps=Installing Python dependencies (this may take 5–20 minutes)...
en.StatusWriteConfig=Saving configuration...

en.ReadyTorchLabel=Torch backend:
en.ReadyFaissLabel=FAISS backend:
en.ReadyLibsLabel=Dependencies folder:
en.ReadyTempLabel=pip temp folder:
en.ReadyTempSystem=(system TEMP)

en.UninstOldLibs=A previous dependencies folder was found. It will be removed before the new installation.

[Files]
Source: "main.py";               DestDir: "{app}"; Flags: ignoreversion
Source: "config.json";           DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "requirements.txt";      DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "install_deps.bat";      DestDir: "{app}"; Flags: ignoreversion
Source: "post_install.bat";      DestDir: "{app}"; Flags: ignoreversion
Source: "run_app.vbs";           DestDir: "{app}"; Flags: ignoreversion
Source: "README.md";             DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "LICENSE";               DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "core\*";    DestDir: "{app}\core";    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "ui\*";      DestDir: "{app}\ui";      Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "utils\*";   DestDir: "{app}\utils";   Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "icons\*";   DestDir: "{app}\icons";   Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "config\*";  DestDir: "{app}\config";  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "models\*";  DestDir: "{app}\models";  Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "python-3.10.0-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist

[Dirs]
Name: "{app}\logs";   Flags: uninsneveruninstall
Name: "{app}\output"; Flags: uninsneveruninstall

[Icons]
Name: "{group}\{#AppName}";          Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Comment: "Launch {#AppName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";    Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Comment: "Launch {#AppName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:TaskDesktop}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "runafter";    Description: "{cm:TaskRunAfter}"; GroupDescription: "{cm:AdditionalIcons}"

[Code]

var
  PageTorch: TInputOptionWizardPage;
  PageFaiss: TInputOptionWizardPage;
  PageLibs: TInputDirWizardPage;
  PageTemp: TInputDirWizardPage;

  FinalLibsPath: String;
  FinalTempPath: String;
  FinalTorchMode: String;
  FinalFaissMode: String;

  PythonExe: String;
  PythonFound: Boolean;


function IsPathSafe(const S: String): Boolean;
var
  i, Code: Integer;
  Ch: Char;
begin
  Result := True;
  for i := 1 to Length(S) do
  begin
    Ch := S[i];
    Code := Ord(Ch);
    if (Code >= $0410) and (Code <= $04FF) then begin Result := False; Exit; end;
    if Ch = ' ' then begin Result := False; Exit; end;
  end;
end;


procedure DetectPython;
begin
  PythonFound := False;
  PythonExe   := '';
  if RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', PythonExe) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.9\InstallPath',  '', PythonExe) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', PythonExe) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.12\InstallPath', '', PythonExe) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', PythonExe) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.9\InstallPath',  '', PythonExe) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', PythonExe) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.12\InstallPath', '', PythonExe) then
  begin
    PythonExe   := AddBackslash(PythonExe) + 'python.exe';
    PythonFound := FileExists(PythonExe);
  end;
  if not PythonFound then
  begin
    PythonExe := ExpandConstant('{sys}') + '\py.exe';
    if FileExists(PythonExe) then PythonFound := True;
  end;
  if not PythonFound then
  begin
    if RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\python.exe', '', PythonExe) then
      PythonFound := FileExists(PythonExe);
  end;
end;


procedure InitializeWizard;
begin
  DetectPython;

  PageTorch := CreateInputOptionPage(
    wpSelectDir,
    ExpandConstant('{cm:PageTorchTitle}'),
    ExpandConstant('{cm:PageTorchSubtitle}'),
    ExpandConstant('{cm:PageTorchSubtitle}'),
    True, False);
  PageTorch.Add(ExpandConstant('{cm:TorchOpt0}'));
  PageTorch.Add(ExpandConstant('{cm:TorchOpt1}'));
  PageTorch.Add(ExpandConstant('{cm:TorchOpt2}'));
  PageTorch.Values[0] := True;

  PageFaiss := CreateInputOptionPage(
    PageTorch.ID,
    ExpandConstant('{cm:PageFaissTitle}'),
    ExpandConstant('{cm:PageFaissSubtitle}'),
    ExpandConstant('{cm:PageFaissSubtitle}'),
    True, False);
  PageFaiss.Add(ExpandConstant('{cm:FaissOpt0}'));
  PageFaiss.Add(ExpandConstant('{cm:FaissOpt1}'));
  PageFaiss.Values[0] := True;

  PageLibs := CreateInputDirPage(
    PageFaiss.ID,
    ExpandConstant('{cm:PageLibsTitle}'),
    ExpandConstant('{cm:PageLibsSubtitle}'),
    ExpandConstant('{cm:LibsPathDesc}') + #13#10 + ExpandConstant('{cm:LibsPathDefault}'),
    False,
    ''
  );
  PageLibs.Add('');
  PageLibs.Values[0] := WizardDirValue + '\pythonlibs';

  PageTemp := CreateInputDirPage(
    PageLibs.ID,
    ExpandConstant('{cm:PageTempTitle}'),
    ExpandConstant('{cm:PageTempSubtitle}'),
    ExpandConstant('{cm:TempPathDesc}') + #13#10 + ExpandConstant('{cm:TempPathDefault}'),
    False,
    ''
  );
  PageTemp.Add('');
  PageTemp.Values[0] := '';
end;

procedure CurPageChanged(CurPageID: Integer);
var
  DefaultLibsPath: String;
begin
  DefaultLibsPath := WizardDirValue + '\pythonlibs';

  if CurPageID = PageLibs.ID then
  begin
    if (Trim(PageLibs.Values[0]) = '') or
       (CompareText(Trim(PageLibs.Values[0]), FinalLibsPath) = 0) or
       (Pos('\pythonlibs', Lowercase(Trim(PageLibs.Values[0]))) > 0) then
    begin
      PageLibs.Values[0] := DefaultLibsPath;
    end;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = PageLibs.ID then
  begin
    if Trim(PageLibs.Values[0]) <> '' then
    begin
      if not IsPathSafe(Trim(PageLibs.Values[0])) then
      begin
        MsgBox(ExpandConstant('{cm:CyrillicWarningLibs}'), mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end;
  end;

  if CurPageID = PageTemp.ID then
  begin
    if Trim(PageTemp.Values[0]) <> '' then
    begin
      if not IsPathSafe(Trim(PageTemp.Values[0])) then
      begin
        if MsgBox(ExpandConstant('{cm:CyrillicWarningTemp}'), mbConfirmation, MB_OKCANCEL) = IDCANCEL then
        begin
          Result := False;
          Exit;
        end;
      end;
    end;
  end;

  if CurPageID = PageFaiss.ID then
  begin
    if PageFaiss.Values[1] then
      MsgBox(ExpandConstant('{cm:FaissGpuWarning}'), mbInformation, MB_OK);
  end;
end;


procedure ResolveSettings;
begin
  if PageTorch.Values[1] then
    FinalTorchMode := 'cu118'
  else if PageTorch.Values[2] then
    FinalTorchMode := 'cu128'
  else
    FinalTorchMode := 'cpu';

  if PageFaiss.Values[1] then
    FinalFaissMode := 'gpu'
  else
    FinalFaissMode := 'cpu';

  if Trim(PageLibs.Values[0]) = '' then
    FinalLibsPath := WizardDirValue + '\pythonlibs'
  else
    FinalLibsPath := Trim(PageLibs.Values[0]);

  if Trim(PageTemp.Values[0]) = '' then
    FinalTempPath := ''
  else
    FinalTempPath := Trim(PageTemp.Values[0]);
end;


function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  S, TorchLabel, FaissLabel, TempVal: String;
begin
  ResolveSettings;
  if FinalTorchMode = 'cu118' then TorchLabel := 'GPU (CUDA 11.8)'
  else if FinalTorchMode = 'cu128' then TorchLabel := 'GPU (CUDA 12.8)'
  else TorchLabel := 'CPU';
  if FinalFaissMode = 'gpu' then FaissLabel := 'GPU (experimental → faiss-cpu)'
  else FaissLabel := 'CPU';
  if FinalTempPath = '' then TempVal := ExpandConstant('{cm:ReadyTempSystem}')
  else TempVal := FinalTempPath;

  S := '';
  if MemoDirInfo <> '' then S := S + MemoDirInfo + NewLine + NewLine;
  S := S
    + Space + ExpandConstant('{cm:ReadyTorchLabel}') + ' ' + TorchLabel      + NewLine
    + Space + ExpandConstant('{cm:ReadyFaissLabel}') + ' ' + FaissLabel      + NewLine
    + Space + ExpandConstant('{cm:ReadyLibsLabel}')  + ' ' + FinalLibsPath   + NewLine
    + Space + ExpandConstant('{cm:ReadyTempLabel}')  + ' ' + TempVal         + NewLine;
  if MemoTasksInfo <> '' then S := S + NewLine + MemoTasksInfo;
  Result := S;
end;


procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode:  Integer;
  BatParams:   String;
  QuotedApp:   String;
  QuotedLibs:  String;
  QuotedTemp:  String;
begin
  if CurStep = ssInstall then
  begin
    ResolveSettings;
    if not PythonFound then
    begin
      if MsgBox(ExpandConstant('{cm:PythonNotFound}'), mbConfirmation, MB_YESNO) = IDYES then
      begin
        WizardForm.StatusLabel.Caption := ExpandConstant('{cm:StatusInstallPython}');
        if FileExists(ExpandConstant('{tmp}') + '\python-3.10.0-amd64.exe') then
        begin
          if not Exec(
            ExpandConstant('{tmp}') + '\python-3.10.0-amd64.exe',
            '/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1',
            '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
            MsgBox('Python installer failed (code ' + IntToStr(ResultCode) + '). Please install Python 3.10 manually.', mbError, MB_OK);
        end else
          MsgBox('Bundled Python installer not found. Please install Python 3.10 from https://www.python.org/', mbError, MB_OK);
      end;
    end;
  end;

  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := ExpandConstant('{cm:StatusWriteConfig}');
    SaveStringToFile(ExpandConstant('{app}') + '\pythonlibs_path.txt', FinalLibsPath, False);

    WizardForm.StatusLabel.Caption := ExpandConstant('{cm:StatusInstallDeps}');
    QuotedApp := ExpandConstant('{app}');
    QuotedLibs := '"' + FinalLibsPath + '"';

    if FinalTempPath <> '' then
      QuotedTemp := '"' + FinalTempPath + '"'
    else
      QuotedTemp := '""';

    BatParams := FinalTorchMode + ' ' + QuotedLibs + ' ' + FinalFaissMode + ' ' + QuotedTemp;

    if not Exec(
      ExpandConstant('{sys}') + '\cmd.exe',
      '/C ""' + QuotedApp + '\post_install.bat" ' + BatParams + '"',
      QuotedApp, SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      MsgBox('Failed to start dependency installation.', mbError, MB_OK);
    end
    else
    begin
      if ResultCode <> 0 then
        MsgBox('Dependency installation failed (exit code ' + IntToStr(ResultCode) + '). Check logs for details.', mbError, MB_OK);
    end;
  end;
end;

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\run_app.vbs"""; WorkingDir: "{app}"; Description: "{cm:TaskRunAfter}"; Flags: postinstall nowait skipifsilent; Tasks: runafter