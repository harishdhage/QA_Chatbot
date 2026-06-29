@echo off
set SCRIPT_DIR=%~dp0
set PYTHON_PATH=%SCRIPT_DIR%My Venv\Scripts\python.exe
set APP_PATH=%SCRIPT_DIR%summarization\appNewTech.py

if not exist "%PYTHON_PATH%" (
  echo Python executable not found at "%PYTHON_PATH%"
  exit /b 1
)
if not exist "%APP_PATH%" (
  echo Streamlit app file not found at "%APP_PATH%"
  exit /b 1
)

echo Launching Streamlit via Python module...
"%PYTHON_PATH%" -m streamlit run "%APP_PATH%"
