@echo off
REM Start all services for multimode_expts in separate Git Bash windows

cd /d "%~dp0"

REM Start nameserver
start "" "C:\Program Files\Git\git-bash.exe" -c "cd '%~dp0' && pixi run python start_nameserver.py; read -p 'Press Enter to close...'"

REM Wait a moment for nameserver to initialize
timeout /t 2 /nobreak >nul

REM Start FastAPI server
start "" "C:\Program Files\Git\git-bash.exe" -c "cd '%~dp0' && pixi run python -m uvicorn job_server.server:app --host 0.0.0.0 --port 8000; read -p 'Press Enter to close...'"

REM Start worker
start "" "C:\Program Files\Git\git-bash.exe" -c "cd '%~dp0' && pixi run python -u -m job_server.worker; read -p 'Press Enter to close...'"

echo All services started in separate windows.
