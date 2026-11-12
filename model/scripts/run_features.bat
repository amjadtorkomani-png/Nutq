@echo off
rem Extract features for train / test_voices / confidence clips. Progress: logs\features.log
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo [%date% %time%] extract features >> logs\features.log
python scripts\extract_features.py --workers 4 >> logs\features.log 2>&1
echo [%date% %time%] FEATURES DONE >> logs\features.log
