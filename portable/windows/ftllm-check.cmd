@echo off
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0libexec\check.py" %*
exit /b %errorlevel%
