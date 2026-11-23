@echo off
REM Usage: check.bat path\to\photo.jpg
cd /d "%~dp0"
.venv\Scripts\python check_house.py %*
