@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] 仮想環境が見つかりません: %PYTHON_EXE%
    echo         先に install_env.bat または requirements.txt に従って環境を作成してください。
    exit /b 1
)

cd /d "%ROOT%"
"%PYTHON_EXE%" main.py %*

endlocal