@echo off
REM Launches the closed_loop mailbox watcher.
REM Polls G:\Shared drives\SLab\Multimode\optimal_control\test\incoming\ for new
REM pulses_*.zip drops and fires batch_runner process_zip on each one (hw mode).
REM
REM Prereq: the job queue server (port 8000) and at least one worker must be running.
REM Validation failures route to failed/<claim_id>/; transient failures retry then
REM poison-pill to processing/. Circuit-breaker pauses after 3 consecutive failures.
REM
REM Double-click this file, or run from a terminal. Ctrl+C to stop cleanly.

setlocal
cd /d "%~dp0..\.."

echo [start_watcher] repo root: %cd%
echo [start_watcher] launching: pixi run python -m job_server.closed_loop.batch_runner watch --mode hw %*
echo [start_watcher] Ctrl+C to stop.
echo.

pixi run python -m job_server.closed_loop.batch_runner watch --mode hw %*

echo.
echo [start_watcher] watcher exited (code %errorlevel%).
pause
