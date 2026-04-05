; ============================================================
;  ParallelFinder — Inno Setup Script
;  Author  : Pozit1vchic
;  Version : v13.0.2.1
;  Style   : modern, bilingual (RU / EN)
; ============================================================

#define AppName        "ParallelFinder"
#define AppVer         "Alpha v13"
#define AppVerNumeric  "13.0.2.1"
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
OutputBaseFilename=ParallelFinder_Setup_Alpha_v13
Compression=lzma2/ultra64
SolidCompression=yes

SetupIconFile=icons\icon.ico
UninstallDisplayIcon={app}\icons\icon.ico

WizardStyle=modern
WizardResizable=no

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
ru.PageTorchDesc=Выберите вариант установки Torch:
ru.PageFaissTitle=Дополнительные компоненты
ru.PageFaissSubtitle=Выберите дополнительные компоненты для установки
ru.PageFaissDesc=Выберите вариант FAISS:
ru.PageLibsTitle=Папка для Python-зависимостей
ru.PageLibsSubtitle=Библиотеки будут установлены сюда, не затрагивая систему
ru.PageTempTitle=Временная папка установки
ru.PageTempSubtitle=Используется pip-ом для кэша и распаковки пакетов

ru.TorchOpt0=CPU  (~1.2 ГБ)  —  нет NVIDIA GPU или не уверены в выборе
ru.TorchOpt1=GPU  (CUDA 11.8, ~3.0 ГБ)  —  старые или слабые видеокарты
ru.TorchOpt2=GPU  (CUDA 12.8, ~3.5 ГБ)  —  новые системы (RTX 40xx/50xx)

ru.FaissOpt0=faiss-cpu  (стабильно, рекомендуется)
ru.FaissOpt1=faiss-gpu  (экспериментально — на Windows может не работать)
ru.FaissGpuWarning=Внимание: FAISS-GPU для Windows пока нестабилен. Будет установлен faiss-cpu.

ru.LibsPathLabel=Папка для Python-библиотек:
ru.LibsPathDesc=Укажите папку, куда будут установлены Python-зависимости. Путь должен содержать только латинские символы. Оставьте пустым для папки по умолчанию:
ru.LibsPathDefault=(по умолчанию: {app}\pythonlibs)

ru.TempPathLabel=Временная папка (для pip):
ru.TempPathDesc=Укажите папку для временных файлов pip. Только латинские символы. Оставьте пустым для системного TEMP.
ru.TempPathDefault=(по умолчанию: системный TEMP)

ru.CyrillicWarningLibs=Папка зависимостей содержит кириллицу или пробелы. Это может привести к сбоям при установке. Используйте путь из латинских символов (например: C:\ParallelFinderLibs).
ru.CyrillicWarningTemp=Временная папка содержит кириллицу или пробелы. Рекомендуется использовать латинский путь (например: C:\Temp\pftemp) или оставить поле пустым.

ru.PythonNotFound=Python не обнаружен. Для работы {#AppName} требуется Python 3.10. Нажмите «Да» для установки поставляемого Python 3.10, или «Нет» для самостоятельной установки (https://www.python.org/).
ru.PythonFoundMsg=Python уже установлен. Повторная установка не требуется.

ru.TaskDesktop=Создать ярлык на Рабочем столе
ru.TaskRunAfter=Запустить {#AppName} после завершения установки

ru.StatusInstallPython=Установка Python 3.10...
ru.StatusInstallDeps=Установка Python-зависимостей (может занять 5–20 минут)...
ru.StatusWriteConfig=Сохранение конфигурации...

ru.ReadyTorchLabel=Torch-бэкенд:
ru.ReadyFaissLabel=FAISS:
ru.ReadyLibsLabel=Папка зависимостей:
ru.ReadyTempLabel=Временная папка pip:
ru.ReadyTempSystem=(системный TEMP)

ru.UninstOldLibs=Обнаружена предыдущая папка зависимостей. Она будет удалена перед новой установкой.

; ---- English ------------------------------------------------
en.PageTorchTitle=Compute Backend — Torch
en.PageTorchSubtitle=Choose how ParallelFinder will run neural network inference
en.PageTorchDesc=Select Torch installation variant:
en.PageFaissTitle=Additional Components
en.PageFaissSubtitle=Choose additional components to install
en.PageFaissDesc=Select FAISS variant:
en.PageLibsTitle=Python Dependencies Folder
en.PageLibsSubtitle=Libraries will be installed here, keeping your system Python clean
en.PageTempTitle=Temporary Installation Folder
en.PageTempSubtitle=Used by pip for package cache and extraction

en.TorchOpt0=CPU  (~1.2 GB)  —  no NVIDIA GPU, or not sure which to pick
en.TorchOpt1=GPU  (CUDA 11.8, ~3.0 GB)  —  older or lower-end GPUs
en.TorchOpt2=GPU  (CUDA 12.8, ~3.5 GB)  —  modern systems (RTX 40xx/50xx)

en.FaissOpt0=faiss-cpu  (stable, recommended)
en.FaissOpt1=faiss-gpu  (experimental — may not work on Windows)
en.FaissGpuWarning=Note: FAISS-GPU for Windows is currently unstable. faiss-cpu will be installed instead.

en.LibsPathLabel=Python libraries folder:
en.LibsPathDesc=Choose where Python dependencies will be installed. ASCII characters only. Leave blank to use the default:
en.LibsPathDefault=(default: {app}\pythonlibs)

en.TempPathLabel=Temporary folder (for pip):
en.TempPathDesc=Specify a folder for pip temporary files. ASCII only. Leave blank to use system TEMP.
en.TempPathDefault=(default: system TEMP)

en.CyrillicWarningLibs=The dependencies folder path contains Cyrillic or spaces, which can break installation. Use an ASCII-only path (e.g. C:\ParallelFinderLibs).
en.CyrillicWarningTemp=The temporary folder path contains Cyrillic or spaces. Use an ASCII-only path (e.g. C:\Temp\pftemp) or leave blank.

en.PythonNotFound=Python was not found. {#AppName} requires Python 3.10. Click Yes to install the bundled Python 3.10, or No to install it yourself (https://www.python.org/).
en.PythonFoundMsg=Python is already installed. No additional Python installation needed.

en.TaskDesktop=Create a Desktop shortcut
en.TaskRunAfter=Launch {#AppName} after setup completes

en.StatusInstallPython=Installing Python 3.10...
en.StatusInstallDeps=Installing Python dependencies (this may take 5–20 minutes)...
en.StatusWriteConfig=Saving configuration...

en.ReadyTorchLabel=Torch backend:
en.ReadyFaissLabel=FAISS:
en.ReadyLibsLabel=Dependencies folder:
en.ReadyTempLabel=pip temp folder:
en.ReadyTempSystem=(system TEMP)

en.UninstOldLibs=A previous dependencies folder was found. It will be removed before the new installation.

[Files]
; ── Основные файлы ───────────────────────────────────────────
Source: "main.py";               DestDir: "{app}"; Flags: ignoreversion
Source: "config.json";           DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "requirements.txt";      DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "install_deps.bat";      DestDir: "{app}"; Flags: ignoreversion
Source: "post_install.bat";      DestDir: "{app}"; Flags: ignoreversion
Source: "run_app.vbs";           DestDir: "{app}"; Flags: ignoreversion
Source: "README.md";             DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "LICENSE";               DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; ── Исходный код ─────────────────────────────────────────────
Source: "core\*";    DestDir: "{app}\core";    Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ui\*";      DestDir: "{app}\ui";      Flags: ignoreversion recursesubdirs createallsubdirs
Source: "utils\*";   DestDir: "{app}\utils";   Flags: ignoreversion recursesubdirs createallsubdirs
Source: "icons\*";   DestDir: "{app}\icons";   Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; ── Python installer (опционально) ───────────────────────────
Source: "python-3.10.0-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist

[Dirs]
Name: "{app}\logs";    Flags: uninsneveruninstall
Name: "{app}\cache";   Flags: uninsneveruninstall
Name: "{app}\models";  Flags: uninsneveruninstall

[Icons]
Name: "{group}\{#AppName}";           Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Comment: "Launch {#AppName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Comment: "Launch {#AppName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:TaskDesktop}";    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "runafter";    Description: "{cm:TaskRunAfter}";   GroupDescription: "{cm:AdditionalIcons}"

[Code]

var
  PageTorch: TInputOptionWizardPage;
  PageFaiss: TInputOptionWizardPage;
  PageLibs:  TInputDirWizardPage;
  PageTemp:  TInputDirWizardPage;

  FinalLibsPath:  String;
  FinalTempPath:  String;
  FinalTorchMode: String;
  FinalFaissMode: String;

  PythonExe:    String;
  PythonFound:  Boolean;


// ── Проверка пути на кириллицу и пробелы ────────────────────
function IsPathSafe(const S: String): Boolean;
var
  i, Code: Integer;
  Ch: Char;
begin
  Result := True;
  for i := 1 to Length(S) do
  begin
    Ch   := S[i];
    Code := Ord(Ch);
    // Кириллица: U+0400..U+04FF
    if (Code >= $0410) and (Code <= $04FF) then
    begin
      Result := False;
      Exit;
    end;
    // Пробел
    if Ch = ' ' then
    begin
      Result := False;
      Exit;
    end;
  end;
end;


// ── Поиск Python в реестре и PATH ───────────────────────────
procedure DetectPython;
var
  RegPath: String;
begin
  PythonFound := False;
  PythonExe   := '';

  // Проверяем HKLM 64-bit и 32-bit через стандартные ветки
  if RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.12\InstallPath', '', RegPath) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', RegPath) or
     RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', RegPath) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.12\InstallPath', '', RegPath) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', RegPath) or
     RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', RegPath) then
  begin
    RegPath := AddBackslash(RegPath) + 'python.exe';
    if FileExists(RegPath) then
    begin
      PythonExe   := RegPath;
      PythonFound := True;
    end;
  end;

  // Проверяем py.exe (Python Launcher for Windows)
  if not PythonFound then
  begin
    RegPath := ExpandConstant('{sys}') + '\py.exe';
    if FileExists(RegPath) then
    begin
      PythonExe   := RegPath;
      PythonFound := True;
    end;
  end;

  // Проверяем App Paths
  if not PythonFound then
  begin
    if RegQueryStringValue(
        HKLM,
        'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\python.exe',
        '', RegPath) then
    begin
      if FileExists(RegPath) then
      begin
        PythonExe   := RegPath;
        PythonFound := True;
      end;
    end;
  end;
end;


// ── Инициализация страниц визарда ───────────────────────────
procedure InitializeWizard;
begin
  DetectPython;

  // Страница выбора Torch-бэкенда
  // Параметры CreateInputOptionPage:
  //   AfterID, Caption, Description, SubCaption, Exclusive, ListBox
  PageTorch := CreateInputOptionPage(
    wpSelectDir,
    ExpandConstant('{cm:PageTorchTitle}'),
    ExpandConstant('{cm:PageTorchSubtitle}'),
    ExpandConstant('{cm:PageTorchDesc}'),
    True, False);
  PageTorch.Add(ExpandConstant('{cm:TorchOpt0}'));
  PageTorch.Add(ExpandConstant('{cm:TorchOpt1}'));
  PageTorch.Add(ExpandConstant('{cm:TorchOpt2}'));
  PageTorch.Values[0] := True;

  // Страница выбора FAISS
  PageFaiss := CreateInputOptionPage(
    PageTorch.ID,
    ExpandConstant('{cm:PageFaissTitle}'),
    ExpandConstant('{cm:PageFaissSubtitle}'),
    ExpandConstant('{cm:PageFaissDesc}'),
    True, False);
  PageFaiss.Add(ExpandConstant('{cm:FaissOpt0}'));
  PageFaiss.Add(ExpandConstant('{cm:FaissOpt1}'));
  PageFaiss.Values[0] := True;

  // Страница выбора папки для библиотек
  PageLibs := CreateInputDirPage(
    PageFaiss.ID,
    ExpandConstant('{cm:PageLibsTitle}'),
    ExpandConstant('{cm:PageLibsSubtitle}'),
    ExpandConstant('{cm:LibsPathDesc}') + #13#10 +
    ExpandConstant('{cm:LibsPathDefault}'),
    False, '');
  PageLibs.Add(ExpandConstant('{cm:LibsPathLabel}'));
  // Значение установим в CurPageChanged, когда WizardDirValue точно известен
  PageLibs.Values[0] := '';

  // Страница выбора временной папки
  PageTemp := CreateInputDirPage(
    PageLibs.ID,
    ExpandConstant('{cm:PageTempTitle}'),
    ExpandConstant('{cm:PageTempSubtitle}'),
    ExpandConstant('{cm:TempPathDesc}') + #13#10 +
    ExpandConstant('{cm:TempPathDefault}'),
    False, '');
  PageTemp.Add(ExpandConstant('{cm:TempPathLabel}'));
  PageTemp.Values[0] := '';
end;


// ── Обновление значений при переходе на страницу ────────────
procedure CurPageChanged(CurPageID: Integer);
begin
  // Устанавливаем дефолтный путь библиотек когда пользователь
  // дошёл до этой страницы (WizardDirValue уже выбран)
  if CurPageID = PageLibs.ID then
  begin
    if Trim(PageLibs.Values[0]) = '' then
      PageLibs.Values[0] := WizardDirValue + '\pythonlibs';
  end;
end;


// ── Валидация при нажатии Next ───────────────────────────────
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = PageLibs.ID then
  begin
    // Если поле НЕ пустое — проверяем безопасность пути
    if (Trim(PageLibs.Values[0]) <> '') and
       (not IsPathSafe(Trim(PageLibs.Values[0]))) then
    begin
      MsgBox(
        ExpandConstant('{cm:CyrillicWarningLibs}'),
        mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;

  if CurPageID = PageTemp.ID then
  begin
    if (Trim(PageTemp.Values[0]) <> '') and
       (not IsPathSafe(Trim(PageTemp.Values[0]))) then
    begin
      // Для temp — предупреждение, не блокируем
      if MsgBox(
          ExpandConstant('{cm:CyrillicWarningTemp}'),
          mbConfirmation, MB_OKCANCEL) = IDCANCEL then
      begin
        Result := False;
        Exit;
      end;
    end;
  end;

  if CurPageID = PageFaiss.ID then
  begin
    if PageFaiss.Values[1] then
      MsgBox(
        ExpandConstant('{cm:FaissGpuWarning}'),
        mbInformation, MB_OK);
  end;
end;


// ── Разрешение финальных настроек ───────────────────────────
procedure ResolveSettings;
begin
  // Torch mode
  if PageTorch.Values[1] then
    FinalTorchMode := 'cu118'
  else if PageTorch.Values[2] then
    FinalTorchMode := 'cu128'
  else
    FinalTorchMode := 'cpu';

  // FAISS mode
  if PageFaiss.Values[1] then
    FinalFaissMode := 'gpu'
  else
    FinalFaissMode := 'cpu';

  // Libs path — если пусто, используем дефолт
  if Trim(PageLibs.Values[0]) = '' then
    FinalLibsPath := WizardDirValue + '\pythonlibs'
  else
    FinalLibsPath := Trim(PageLibs.Values[0]);

  // Temp path — пустая строка = системный TEMP
  if Trim(PageTemp.Values[0]) = '' then
    FinalTempPath := ''
  else
    FinalTempPath := Trim(PageTemp.Values[0]);
end;


// ── Страница "Готово к установке" ───────────────────────────
function UpdateReadyMemo(
  Space, NewLine,
  MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo,
  MemoGroupInfo, MemoTasksInfo: String): String;
var
  S, TorchLabel, FaissLabel, TempVal: String;
begin
  ResolveSettings;

  if FinalTorchMode = 'cu118' then
    TorchLabel := 'GPU (CUDA 11.8)'
  else if FinalTorchMode = 'cu128' then
    TorchLabel := 'GPU (CUDA 12.8)'
  else
    TorchLabel := 'CPU';

  if FinalFaissMode = 'gpu' then
    FaissLabel := 'GPU (experimental → faiss-cpu)'
  else
    FaissLabel := 'CPU';

  if FinalTempPath = '' then
    TempVal := ExpandConstant('{cm:ReadyTempSystem}')
  else
    TempVal := FinalTempPath;

  S := '';

  if MemoDirInfo <> '' then
    S := S + MemoDirInfo + NewLine + NewLine;

  S := S
    + Space + ExpandConstant('{cm:ReadyTorchLabel}')
    + ' ' + TorchLabel    + NewLine
    + Space + ExpandConstant('{cm:ReadyFaissLabel}')
    + ' ' + FaissLabel    + NewLine
    + Space + ExpandConstant('{cm:ReadyLibsLabel}')
    + ' ' + FinalLibsPath + NewLine
    + Space + ExpandConstant('{cm:ReadyTempLabel}')
    + ' ' + TempVal       + NewLine;

  if MemoGroupInfo <> '' then
    S := S + NewLine + MemoGroupInfo;

  if MemoTasksInfo <> '' then
    S := S + NewLine + MemoTasksInfo;

  Result := S;
end;


// ── Основная логика шагов установки ─────────────────────────
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode:  Integer;
  BatPath:     String;
  BatParams:   String;
  QuotedApp:   String;
  QuotedLibs:  String;
  QuotedTemp:  String;
  CmdExe:      String;
begin
  // ssInstall: перед копированием файлов — установка Python если нужно
  if CurStep = ssInstall then
  begin
    ResolveSettings;

    if not PythonFound then
    begin
      if MsgBox(
          ExpandConstant('{cm:PythonNotFound}'),
          mbConfirmation, MB_YESNO) = IDYES then
      begin
        Log('Installing bundled Python 3.10...');
        if FileExists(ExpandConstant('{tmp}') + '\python-3.10.0-amd64.exe') then
        begin
          if not Exec(
              ExpandConstant('{tmp}') + '\python-3.10.0-amd64.exe',
              '/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1',
              '',
              SW_HIDE,
              ewWaitUntilTerminated,
              ResultCode) then
          begin
            MsgBox(
              'Python installer failed to start.' + #13#10 +
              'Please install Python 3.10 manually from https://www.python.org/',
              mbError, MB_OK);
          end else if ResultCode <> 0 then
          begin
            MsgBox(
              'Python installer returned error code ' +
              IntToStr(ResultCode) + '.' + #13#10 +
              'Please install Python 3.10 manually from https://www.python.org/',
              mbError, MB_OK);
          end;
        end else
        begin
          MsgBox(
            'Bundled Python installer not found.' + #13#10 +
            'Please install Python 3.10 from https://www.python.org/',
            mbError, MB_OK);
        end;
      end;
    end;
  end;

  // ssPostInstall: после копирования файлов — запуск установки зависимостей
  if CurStep = ssPostInstall then
  begin
    ResolveSettings;

    // Записываем pythonlibs_path.txt
    Log('Writing pythonlibs_path.txt: ' + FinalLibsPath);
    SaveStringToFile(
      ExpandConstant('{app}') + '\pythonlibs_path.txt',
      FinalLibsPath, False);

    Log('Starting dependency installation...');
    Log('  Torch mode : ' + FinalTorchMode);
    Log('  FAISS mode : ' + FinalFaissMode);
    Log('  Libs path  : ' + FinalLibsPath);
    Log('  Temp path  : ' + FinalTempPath);

    QuotedApp  := ExpandConstant('{app}');
    BatPath    := QuotedApp + '\post_install.bat';
    CmdExe     := ExpandConstant('{sys}') + '\cmd.exe';

    // Формируем аргументы:
    //   arg1 = torch mode (без кавычек, латиница без пробелов)
    //   arg2 = libs path  (в кавычках)
    //   arg3 = faiss mode (без кавычек, латиница без пробелов)
    //   arg4 = temp path  (в кавычках или "")
    QuotedLibs := '"' + FinalLibsPath + '"';

    if FinalTempPath <> '' then
      QuotedTemp := '"' + FinalTempPath + '"'
    else
      QuotedTemp := '""';

    BatParams := FinalTorchMode + ' ' +
                 QuotedLibs    + ' ' +
                 FinalFaissMode + ' ' +
                 QuotedTemp;

    // Итоговая строка для cmd: /C ""%BatPath%" %BatParams%"
    // Двойные кавычки снаружи обязательны для cmd /C с кавычками внутри
    if not Exec(
        CmdExe,
        '/C ""' + BatPath + '" ' + BatParams + '"',
        QuotedApp,
        SW_HIDE,
        ewWaitUntilTerminated,
        ResultCode) then
    begin
      MsgBox(
        'Failed to start post_install.bat.' + #13#10 +
        'Please run it manually from:' + #13#10 +
        BatPath,
        mbError, MB_OK);
    end else
    begin
      if ResultCode <> 0 then
        MsgBox(
          'Dependency installation failed (exit code ' +
          IntToStr(ResultCode) + ').' + #13#10 +
          'Check the log file in: ' + QuotedApp + '\logs\',
          mbError, MB_OK)
      else
        Log('Dependencies installed successfully.');
    end;
  end;
end;

[Run]
; Запуск приложения после установки (только если выбрана задача runafter)
Filename: "{sys}\wscript.exe"; \
  Parameters: """{app}\run_app.vbs"""; \
  WorkingDir: "{app}"; \
  Description: "{cm:TaskRunAfter}"; \
  Flags: postinstall nowait skipifsilent; \
  Tasks: runafter