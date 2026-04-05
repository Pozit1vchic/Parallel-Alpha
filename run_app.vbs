' ============================================================
'  ParallelFinder Alpha v13 — run_app.vbs
'
'  Launches main.py via pythonw (no console window).
'  Place this file in the same folder as main.py.
' ============================================================

Option Explicit

Dim oShell, oFSO
Dim appDir, pythonwExe, mainScript
Dim libsPathFile, libsPath
Dim cmdLine

Set oShell = CreateObject("WScript.Shell")
Set oFSO   = CreateObject("Scripting.FileSystemObject")

' ── Resolve app directory ─────────────────────────────────────
appDir = oFSO.GetParentFolderName(WScript.ScriptFullName)

' ── Read pythonlibs path ──────────────────────────────────────
libsPath     = ""
libsPathFile = appDir & "\pythonlibs_path.txt"

If oFSO.FileExists(libsPathFile) Then
    Dim fStream
    Set fStream = oFSO.OpenTextFile(libsPathFile, 1)
    If Not fStream.AtEndOfStream Then
        libsPath = Trim(fStream.ReadLine())
    End If
    fStream.Close
    Set fStream = Nothing
End If

' ── Locate pythonw.exe ────────────────────────────────────────
pythonwExe = ""

' Пытаемся найти python через py-launcher, получаем путь к exe
Dim tmpFile, runResult
tmpFile = appDir & "\~pypath.tmp"

' Запускаем cmd синхронно (True) чтобы дождаться результата
On Error Resume Next
runResult = oShell.Run( _
    "cmd.exe /C py -3 -c ""import sys; print(sys.executable)"" > """ & tmpFile & """", _
    0, True)
On Error GoTo 0

If oFSO.FileExists(tmpFile) Then
    Dim fs2, rawPath, pyDir
    Set fs2 = oFSO.OpenTextFile(tmpFile, 1)
    If Not fs2.AtEndOfStream Then
        rawPath = Trim(fs2.ReadLine())
    End If
    fs2.Close
    Set fs2 = Nothing

    ' Удаляем временный файл
    On Error Resume Next
    oFSO.DeleteFile tmpFile, True
    On Error GoTo 0

    ' Ищем pythonw.exe рядом с python.exe
    If rawPath <> "" Then
        pyDir = oFSO.GetParentFolderName(rawPath)
        If oFSO.FileExists(pyDir & "\pythonw.exe") Then
            pythonwExe = pyDir & "\pythonw.exe"
        End If
    End If
End If

' Если py launcher не помог — пробуем реестр
If pythonwExe = "" Then
    Dim oReg, regPath, regVal
    Set oReg = CreateObject("WScript.Shell")

    ' Пробуем стандартные пути в реестре через environ (упрощённый вариант)
    ' Основной fallback — просто "pythonw" (должен быть в PATH)
    pythonwExe = "pythonw"
End If

' ── Verify main.py exists ─────────────────────────────────────
mainScript = appDir & "\main.py"

If Not oFSO.FileExists(mainScript) Then
    MsgBox "Cannot find main.py in:" & vbCrLf & appDir & vbCrLf & vbCrLf & _
           "Please reinstall ParallelFinder.", _
           vbCritical Or vbOKOnly, "ParallelFinder — Launch Error"
    WScript.Quit 1
End If

' ── Устанавливаем PYTHONPATH если нашли папку с библиотеками ──
If libsPath <> "" Then
    ' WshEnvironment — правильный способ изменить переменную окружения процесса
    Dim oEnv
    Set oEnv = oShell.Environment("PROCESS")

    Dim currentPythonPath
    currentPythonPath = oEnv("PYTHONPATH")

    If currentPythonPath <> "" Then
        oEnv("PYTHONPATH") = libsPath & ";" & currentPythonPath
    Else
        oEnv("PYTHONPATH") = libsPath
    End If

    Set oEnv = Nothing
End If

' ── Build command line ────────────────────────────────────────
If pythonwExe = "pythonw" Then
    ' pythonw в PATH — без кавычек вокруг имени интерпретатора
    cmdLine = "pythonw """ & mainScript & """"
Else
    ' Полный путь — берём в кавычки
    cmdLine = """" & pythonwExe & """ """ & mainScript & """"
End If

' ── Launch (0 = скрытое окно, False = не ждать завершения) ───
oShell.CurrentDirectory = appDir
oShell.Run cmdLine, 0, False

' ── Cleanup ──────────────────────────────────────────────────
Set oFSO   = Nothing
Set oShell = Nothing

WScript.Quit 0