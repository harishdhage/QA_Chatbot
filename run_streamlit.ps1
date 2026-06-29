$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot 'My Venv\Scripts\python.exe'
$appPath = Join-Path $projectRoot 'summarization\appNewTech.py'

if (-not (Test-Path $pythonPath)) {
    Write-Error "Python executable not found at: $pythonPath"
    exit 1
}

if (-not (Test-Path $appPath)) {
    Write-Error "Streamlit app file not found at: $appPath"
    exit 1
}

Write-Host "Launching Streamlit via Python module..."
& $pythonPath -m streamlit run $appPath
