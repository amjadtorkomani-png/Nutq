@echo off
rem سامع النطق: starts the local app, then open http://127.0.0.1:8765 in Chrome or Edge
cd /d "%~dp0"
start "" http://127.0.0.1:8765
python scripts\app.py
pause
