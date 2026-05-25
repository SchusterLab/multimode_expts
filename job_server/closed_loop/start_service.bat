@echo off
REM Launches the closed_loop QOC service (FastAPI on 127.0.0.1:18765).
REM Prereq: the job queue server (port 8000) and at least one worker must be running.
REM Double-click this file, or run from a terminal.

setlocal
cd /d "%~dp0..\.."

echo [start_service] repo root: %cd%
echo [start_service] launching: pixi run python -m job_server.closed_loop.service %*
echo.

pixi run python -m job_server.closed_loop.service %*

echo.
echo [start_service] service exited (code %errorlevel%).
pause
