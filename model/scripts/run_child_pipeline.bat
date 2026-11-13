@echo off
rem Finish the child-voice conversion and check it. Runs on its own, so it keeps
rem going if the Claude session restarts. Progress: logs\pipeline.log
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo [%date% %time%] childify (resume) >> logs\pipeline.log
python scripts\childify.py --children-per-voice 4 --workers 16 >> logs\pipeline.log 2>&1

echo [%date% %time%] check + delete broken files >> logs\pipeline.log
python scripts\check_child.py --fix --workers 8 >> logs\pipeline.log 2>&1

echo [%date% %time%] childify (re-make deleted files) >> logs\pipeline.log
python scripts\childify.py --children-per-voice 4 --workers 16 >> logs\pipeline.log 2>&1

echo [%date% %time%] final check >> logs\pipeline.log
python scripts\check_child.py --recheck --workers 8 >> logs\pipeline.log 2>&1

echo [%date% %time%] PIPELINE DONE >> logs\pipeline.log
