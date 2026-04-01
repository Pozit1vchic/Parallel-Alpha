#define MyAppName "ParallelFinder"
#define MyAppVersion "v12.5"
#define MyAppVersionNumeric "12.5.0.0"
#define MyAppPublisher "Pozit1vchic"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoCompany={#MyAppPublisher}
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersionNumeric}
VersionInfoVersion={#MyAppVersionNumeric}
VersionInfoDescription={#MyAppName} Installer
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=output
OutputBaseFilename=ParallelFinderSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=yes
PrivilegesRequired=admin
DisableDirPage=no
SetupIconFile=icons\icon.ico
UninstallDisplayIcon={app}\icons\icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Files]
Source: "main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "config.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_deps.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "post_install.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "run_app.vbs"; DestDir: "{app}"; Flags: ignoreversion
Source: "python-3.10.0-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

Source: "core\*"; DestDir: "{app}\core"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "utils\*"; DestDir: "{app}\utils"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "icons\*"; DestDir: "{app}\icons"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "config\*"; DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
Name: "{app}\logs"
Name: "{app}\output"

[Tasks]
Name: "startmenuicon"; Description: "{code:GetStartMenuTaskText}"; GroupDescription: "{code:GetAdditionalTasksText}"; Flags: checkedonce
Name: "desktopicon"; Description: "{code:GetDesktopTaskText}"; GroupDescription: "{code:GetAdditionalTasksText}"; Flags: unchecked
Name: "launchafterinstall"; Description: "{code:GetLaunchAfterInstallTaskText}"; GroupDescription: "{code:GetAdditionalTasksText}"; Flags: checkedonce

[Icons]
Name: "{group}\ParallelFinder"; Filename: "{app}\run_app.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Tasks: startmenuicon
Name: "{autodesktop}\ParallelFinder"; Filename: "{app}\run_app.vbs"; WorkingDir: "{app}"; IconFilename: "{app}\icons\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{tmp}\python-3.10.0-amd64.exe"; Parameters: "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0"; StatusMsg: "{code:GetPythonInstallStatusMsg}"; Flags: waituntilterminated; Check: not IsPythonInstalled
Filename: "{cmd}"; Parameters: "/c """"{app}\post_install.bat"" ""{code:GetTorchMode}"" ""{code:GetLibsPath}"" ""{code:GetFaissMode}"" ""{code:GetTempPath}"""""; StatusMsg: "{code:GetInstallStatus}"; Flags: waituntilterminated runhidden
Filename: "{app}\run_app.vbs"; Description: "{code:GetLaunchNowText}"; Flags: nowait postinstall skipifsilent; Tasks: launchafterinstall

[Code]
var
  TorchPage: TInputOptionWizardPage;
  FaissPage: TInputOptionWizardPage;
  LibsPathPage: TInputDirWizardPage;
  TempPathPage: TInputDirWizardPage;

function IsRussian(): Boolean;
begin
  Result := ActiveLanguage = 'russian';
end;

function IsPythonInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result :=
    Exec('cmd.exe', '/C python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);

  if not Result then
    Result :=
      Exec('cmd.exe', '/C py --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function InstallPythonIfNeeded(): Boolean;
var
  ResultCode: Integer;
  MsgText: String;
begin
  Result := True;

  if not IsPythonInstalled() then
  begin
    if IsRussian() then
      MsgText := 'Python не найден.' + #13#10 +
                 'Установщик может установить Python автоматически.' + #13#10#13#10 +
                 'Продолжить?'
    else
      MsgText := 'Python was not found.' + #13#10 +
                 'The installer can install Python automatically.' + #13#10#13#10 +
                 'Continue?';

    if MsgBox(MsgText, mbConfirmation, MB_YESNO) = IDYES then
    begin
      if not Exec(
        ExpandConstant('{tmp}\python-3.10.0-amd64.exe'),
        '/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0',
        '',
        SW_SHOW,
        ewWaitUntilTerminated,
        ResultCode
      ) then
      begin
        if IsRussian() then
          MsgBox('Не удалось запустить установщик Python.', mbError, MB_OK)
        else
          MsgBox('Failed to launch Python installer.', mbError, MB_OK);
        Result := False;
        exit;
      end;

      if not IsPythonInstalled() then
      begin
        if IsRussian() then
          MsgBox('Python не был найден после установки. Установите его вручную и повторите попытку.', mbError, MB_OK)
        else
          MsgBox('Python was not detected after installation. Please install it manually and try again.', mbError, MB_OK);
        Result := False;
        exit;
      end;
    end
    else
    begin
      Result := False;
      exit;
    end;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = wpReady then
    Result := InstallPythonIfNeeded();
end;

procedure InitializeWizard;
begin
  if IsRussian() then
  begin
    TorchPage := CreateInputOptionPage(
      wpSelectDir,
      'Выбор Torch',
      'Выберите версию PyTorch',
      'CPU подходит, если у вас нет NVIDIA GPU. GPU-варианты требуют совместимую NVIDIA-видеокарту и драйверы.',
      True,
      False
    );
    TorchPage.Add('CPU (~1.2 GB) - если нет NVIDIA GPU или вы не уверены');
    TorchPage.Add('GPU (CUDA 11.8, ~3.0 GB) - для более слабых или старых систем/видеокарт');
    TorchPage.Add('GPU (CUDA 12.8, ~3.5 GB) - для новых систем, драйверов и видеокарт');
    TorchPage.SelectedValueIndex := 0;

    FaissPage := CreateInputOptionPage(
      TorchPage.ID,
      'Выбор FAISS',
      'Выберите версию FAISS',
      'FAISS GPU на Windows может работать нестабильно. Рекомендуется CPU.',
      True,
      False
    );
    FaissPage.Add('CPU (стабильно, рекомендуется)');
    FaissPage.Add('GPU (экспериментально, на Windows может не работать)');
    FaissPage.SelectedValueIndex := 0;

    LibsPathPage := CreateInputDirPage(
      FaissPage.ID,
      'Папка для зависимостей',
      'Выберите папку для установки Python-зависимостей',
      'Если оставить поле пустым, будет использована стандартная папка внутри приложения.',
      False,
      ''
    );
    LibsPathPage.Add('');
    LibsPathPage.Values[0] := '';

    TempPathPage := CreateInputDirPage(
      LibsPathPage.ID,
      'Папка TEMP для установки',
      'Выберите временную папку для pip/распаковки',
      'Если оставить поле пустым, будет использован системный TEMP Windows.',
      False,
      ''
    );
    TempPathPage.Add('');
    TempPathPage.Values[0] := '';
  end
  else
  begin
    TorchPage := CreateInputOptionPage(
      wpSelectDir,
      'Torch Backend',
      'Choose PyTorch version',
      'CPU is suitable if you do not have an NVIDIA GPU. GPU options require compatible NVIDIA hardware and drivers.',
      True,
      False
    );
    TorchPage.Add('CPU (~1.2 GB) - if you do not have an NVIDIA GPU or are not sure');
    TorchPage.Add('GPU (CUDA 11.8, ~3.0 GB) - for older or weaker systems/GPUs');
    TorchPage.Add('GPU (CUDA 12.8, ~3.5 GB) - for newer systems, drivers and GPUs');
    TorchPage.SelectedValueIndex := 0;

    FaissPage := CreateInputOptionPage(
      TorchPage.ID,
      'FAISS Backend',
      'Choose FAISS version',
      'FAISS GPU on Windows may be unstable. CPU is recommended.',
      True,
      False
    );
    FaissPage.Add('CPU (stable, recommended)');
    FaissPage.Add('GPU (experimental, may not work on Windows)');
    FaissPage.SelectedValueIndex := 0;

    LibsPathPage := CreateInputDirPage(
      FaissPage.ID,
      'Dependencies folder',
      'Choose where Python dependencies will be installed',
      'If left empty, the default folder inside the application directory will be used.',
      False,
      ''
    );
    LibsPathPage.Add('');
    LibsPathPage.Values[0] := '';

    TempPathPage := CreateInputDirPage(
      LibsPathPage.ID,
      'TEMP folder for installation',
      'Choose a temporary folder for pip/extraction',
      'If left empty, the default Windows system TEMP folder will be used.',
      False,
      ''
    );
    TempPathPage.Add('');
    TempPathPage.Values[0] := '';
  end;
end;

function GetTorchMode(Param: String): String;
begin
  case TorchPage.SelectedValueIndex of
    1: Result := 'cu118';
    2: Result := 'cu128';
  else
    Result := 'cpu';
  end;
end;

function GetFaissMode(Param: String): String;
begin
  case FaissPage.SelectedValueIndex of
    1: Result := 'gpu';
  else
    Result := 'cpu';
  end;
end;

function GetLibsPath(Param: String): String;
begin
  Result := Trim(LibsPathPage.Values[0]);
  if Result = '' then
    Result := ExpandConstant('{app}\pythonlibs');
end;

function GetTempPath(Param: String): String;
begin
  Result := Trim(TempPathPage.Values[0]);
end;

function GetInstallStatus(Param: String): String;
begin
  if IsRussian() then
    Result := 'Устанавливаются Python-зависимости...'
  else
    Result := 'Installing Python dependencies...';
end;

function GetPythonInstallStatusMsg(Param: String): String;
begin
  if IsRussian() then
    Result := 'Установка Python 3.10...'
  else
    Result := 'Installing Python 3.10...';
end;

function GetAdditionalTasksText(Param: String): String;
begin
  if IsRussian() then
    Result := 'Дополнительные задачи:'
  else
    Result := 'Additional tasks:';
end;

function GetStartMenuTaskText(Param: String): String;
begin
  if IsRussian() then
    Result := 'Создать ярлык в меню Пуск'
  else
    Result := 'Create Start Menu shortcut';
end;

function GetDesktopTaskText(Param: String): String;
begin
  if IsRussian() then
    Result := 'Создать ярлык на рабочем столе'
  else
    Result := 'Create Desktop shortcut';
end;

function GetLaunchAfterInstallTaskText(Param: String): String;
begin
  if IsRussian() then
    Result := 'Запустить ParallelFinder после установки'
  else
    Result := 'Launch ParallelFinder after installation';
end;

function GetLaunchNowText(Param: String): String;
begin
  if IsRussian() then
    Result := 'Запустить ParallelFinder'
  else
    Result := 'Launch ParallelFinder';
end;