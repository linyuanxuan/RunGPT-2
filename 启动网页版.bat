@echo off
setlocal
title GPT-2 Local Server
set "PY=D:\GPT2Local\venv\Scripts\python.exe"
cd /d "%~dp0"

if not exist "%PY%" (
  echo [ERROR] Python venv not found: %PY%
  pause
  exit /b 1
)

echo ==========================================================
echo   GPT-2 (124M) local inference - CPU
echo   Model : D:\GPT2Local\models\gpt2
echo   Open  : http://127.0.0.1:7860
echo   Stop  : close this window or press Ctrl+C
echo ==========================================================
echo.

"%PY%" app.py

echo.
echo Server stopped.
pause
