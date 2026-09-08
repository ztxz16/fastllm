@echo off
"%~dp0runtime\python.exe" -I -B -X utf8 %*
exit /b %errorlevel%
