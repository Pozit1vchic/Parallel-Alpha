#define MyAppName "ParallelFinder"
#define MyAppVersion "12.5 Alpha"
#define MyAppPublisher "Pozit1vchic"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=output
OutputBaseFilename=ParallelFinderSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Files]
Source: "main.py"; DestDir: "{app}"
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "config.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_deps.bat"; DestDir: "{app}"
Source: "post_install.bat"; DestDir: "{app}"
Source: "run_app.vbs"; DestDir: "{app}"
Source: "python-3.10.0-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

Source: "core\*"; DestDir: "{app}\core"; Flags: recursesubdirs createallsubdirs
Source: "ui\*"; DestDir: "{app}\ui"; Flags: recursesubdirs createallsubdirs
Source: "utils\*"; DestDir: "{app}\utils"; Flags: recursesubdirs createallsubdirs
Source: "icons\*"; DestDir: "{app}\icons"; Flags: recursesubdirs createallsubdirs

[Tasks]
Name: "startmenuicon"; Description: "{code:GetStartMenuTaskText}"; GroupDescription: "{code:GetAdditionalTasksText}"; Flags: unchecked
Name: "desktopicon"; Description: "{code:GetDesktopTaskText}"; GroupDescription: "{code:GetAdditionalTasksText}"; Flags: checkedonce

[Icons]
Name: "{group}\ParallelFinder"; Filename: "{app}\run_app.vbs"; IconFilename: "{app}\icons\icon.ico"; Tasks: startmenuicon
Name: "{autodesktop}\ParallelFinder"; Filename: "{app}\run_app.vbs"; IconFilename: "{app}\icons\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\post_install.bat"; Parameters: "{code:GetInstallMode} ""{code:GetLibsPath}"""; Flags: waituntilterminated; StatusMsg: "{code:GetInstallStatus}"

[Code]
var
  InstallModePage: TInputOptionWizardPage;
  LibsPathPage: TInputDirWizardPage;

function IsPythonInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/C python --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);

  if not Result then
    Result := Exec('cmd.exe', '/C py --version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

function InstallPythonIfNeeded(): Boolean;
var
  ResultCode: Integer;
  MsgText: String;
begin
  Result := True;

  if not IsPythonInstalled() then
  begin
    if ActiveLanguage = 'russian' then
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
        '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0',
        '',
        SW_SHOW,
        ewWaitUntilTerminated,
        ResultCode
      ) then
      begin
        if ActiveLanguage = 'russian' then
          MsgBox('Не удалось запустить установщик Python.', mbError, MB_OK)
        else
          MsgBox('Failed to launch Python installer.', mbError, MB_OK);
        Result := False;
        exit;
      end;

      if not IsPythonInstalled() then
      begin
        if ActiveLanguage = 'russian' then
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
  if ActiveLanguage = 'russian' then
  begin
    InstallModePage := CreateInputOptionPage(
      wpSelectDir,
      'Выбор режима установки',
      'Выберите версию библиотек',
      'Выберите CPU или подходящую GPU CUDA-версию',
      True,
      False
    );

    InstallModePage.Add('CPU (~1.2 GB) - если нет NVIDIA GPU или вы не уверены');
    InstallModePage.Add('GPU CUDA 11.8 (~3.0 GB) - для более старых NVIDIA-систем и лучшей совместимости');
    InstallModePage.Add('GPU CUDA 12.8 (~3.5 GB) - для более новых Windows-систем, свежих драйверов NVIDIA и современных GPU');
  end
  else
  begin
    InstallModePage := CreateInputOptionPage(
      wpSelectDir,
      'Installation mode',
      'Choose library version',
      'Choose CPU or a suitable GPU CUDA version',
      True,
      False
    );

    InstallModePage.Add('CPU (~1.2 GB) - if you do not have an NVIDIA GPU or are not sure');
    InstallModePage.Add('GPU CUDA 11.8 (~3.0 GB) - for older NVIDIA systems and wider compatibility');
    InstallModePage.Add('GPU CUDA 12.8 (~3.5 GB) - for newer Windows systems, newer NVIDIA drivers, and modern GPUs');
  end;

  InstallModePage.SelectedValueIndex := 0;

  if ActiveLanguage = 'russian' then
  begin
    LibsPathPage := CreateInputDirPage(
      InstallModePage.ID,
      'Папка для Python-библиотек',
      'Выберите папку для установки библиотек',
      'Например: D:\ParallelFinderLibs',
      False,
      ''
    );
  end
  else
  begin
    LibsPathPage := CreateInputDirPage(
      InstallModePage.ID,
      'Folder for Python libraries',
      'Choose where Python libraries will be installed',
      'Example: D:\ParallelFinderLibs',
      False,
      ''
    );
  end;

  LibsPathPage.Add('');
  LibsPathPage.Values[0] := 'D:\ParallelFinderLibs';
end;

function GetInstallMode(Param: String): String;
begin
  case InstallModePage.SelectedValueIndex of
    1: Result := 'cu118';
    2: Result := 'cu128';
  else
    Result := 'cpu';
  end;
end;

function GetLibsPath(Param: String): String;
begin
  Result := LibsPathPage.Values[0];
end;

function GetInstallStatus(Param: String): String;
begin
  if ActiveLanguage = 'russian' then
    Result := 'Устанавливаются Python-библиотеки...'
  else
    Result := 'Installing Python dependencies...';
end;

function GetAdditionalTasksText(Param: String): String;
begin
  if ActiveLanguage = 'russian' then
    Result := 'Дополнительные задачи:'
  else
    Result := 'Additional tasks:';
end;

function GetStartMenuTaskText(Param: String): String;
begin
  if ActiveLanguage = 'russian' then
    Result := 'Создать ярлык в меню Пуск'
  else
    Result := 'Create Start Menu shortcut';
end;

function GetDesktopTaskText(Param: String): String;
begin
  if ActiveLanguage = 'russian' then
    Result := 'Создать ярлык на рабочем столе'
  else
    Result := 'Create Desktop shortcut';
end;