' ============================================================
'  ParallelFinder — run_app.vbs
'
'  Launches main.py via pythonw (no console window).
'  Place this file in the same folder as main.py.
'
'  The shortcut created by the installer points to this file.
'  WorkingDir of the shortcut must be set to the app folder.
' ============================================================

Option Explicit

Dim oShell, oFSO
Dim appDir, pythonwExe, mainScript
Dim cmdLine

Set oShell = CreateObject("WScript.Shell")
Set oFSO   = CreateObject("Scripting.FileSystemObject")

' ── Resolve app directory ─────────────────────────────────────
' WScript.ScriptFullName gives the full path to this .vbs file.
' The app folder is its parent directory.
appDir = oFSO.GetParentFolderName(WScript.ScriptFullName)

' ── Locate pythonw.exe ────────────────────────────────────────
' We try several strategies in order of preference.
pythonwExe = ""

' 1. Try "py -3" launcher path → derive pythonw from it
Dim pyLauncherResult
On Error Resume Next
pyLauncherResult = oShell.Run("cmd.exe /C ""py -3 -c """"import sys; print(sys.executable)"""" > " & """" & appDir & "\~pypath.tmp" & """" & """", 0, True)
On Error GoTo 0

If oFSO.FileExists(appDir & "\~pypath.tmp") Then
    Dim fStream
    Set fStream = oFSO.OpenTextFile(appDir & "\~pypath.tmp", 1)
    If Not fStream.AtEndOfStream Then
        Dim rawPath
        rawPath = Trim(fStream.ReadLine())
        ' rawPath is python.exe — replace with pythonw.exe in same folder
        Dim pyDir
        pyDir = oFSO.GetParentFolderName(rawPath)
        If oFSO.FileExists(pyDir & "\pythonw.exe") Then
            pythonwExe = pyDir & "\pythonw.exe"
        End If
    End If
    fStream.Close
    ' Clean up temp file
    On Error Resume Next
    oFSO.DeleteFile appDir & "\~pypath.tmp", True
    On Error GoTo 0
End If

' 2. Fallback: try "pythonw" directly from PATH
If pythonwExe = "" Then
    pythonwExe = "pythonw"
End If

' ── Build main script path ────────────────────────────────────
mainScript = appDir & "\main.py"

' Verify main.py exists
If Not oFSO.FileExists(mainScript) Then
    MsgBox "Cannot find main.py in:" & vbCrLf & appDir & vbCrLf & vbCrLf & _
           "Please reinstall ParallelFinder.", _
           vbCritical Or vbOKOnly, "ParallelFinder — Launch Error"
    WScript.Quit 1
End If

' ── Build command line ────────────────────────────────────────
' Quote both the executable and the script path for safety.
If pythonwExe = "pythonw" Then
    ' Using plain command — no extra quoting needed for exe
    cmdLine = "pythonw """ & mainScript & """"
Else
    cmdLine = """" & pythonwExe & """ """ & mainScript & """"
End If

' ── Launch ───────────────────────────────────────────────────
' oShell.Run:
'   arg1 = command
'   arg2 = window style: 0 = hidden
'   arg3 = bWaitOnReturn: False = fire and forget
oShell.CurrentDirectory = appDir
oShell.Run cmdLine, 0, False

' ── Cleanup ──────────────────────────────────────────────────
Set oFSO   = Nothing
Set oShell = Nothing