@echo off
setlocal
"%~dp0ftllm.exe" launch %*
if errorlevel 1 pause
