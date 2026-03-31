Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

currentDir = FSO.GetParentFolderName(WScript.ScriptFullName)
mainPy = """" & currentDir & "\main.py" & """"

WshShell.CurrentDirectory = currentDir

On Error Resume Next
WshShell.Run "pythonw " & mainPy, 0, False
If Err.Number <> 0 Then
    MsgBox "Python was not found. Please reinstall the application or install Python manually.", 16, "ParallelFinder"
End If