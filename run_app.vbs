Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

currentDir = FSO.GetParentFolderName(WScript.ScriptFullName)
mainPy = """" & currentDir & "\main.py" & """"

WshShell.CurrentDirectory = currentDir
WshShell.Run "cmd /k python " & mainPy, 1, False