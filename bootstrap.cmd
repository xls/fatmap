@echo off
rem fatmap bootstrap (Windows). See bootstrap.py --help.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (py -3 bootstrap.py %*) else (python bootstrap.py %*)
