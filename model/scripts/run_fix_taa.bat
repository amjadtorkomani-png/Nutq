@echo off
rem Re-make the five taa-marbuta words (كرة شجرة سيارة طائرة بقرة) with the final ة written as ه,
rem then their blends and child versions, then check. Progress: logs\fix_taa.log
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set WORDS=w12,w13,w14,w15,w16

echo [%date% %time%] TTS for %WORDS% >> logs\fix_taa.log
python scripts\generate_tts.py --engines edge --multilingual --rates "+0%%,-20%%" --errors-per-word 4 --word-ids %WORDS% --overwrite >> logs\fix_taa.log 2>&1

echo [%date% %time%] blends for %WORDS% >> logs\fix_taa.log
python scripts\make_blends.py --word-ids %WORDS% --workers 4 >> logs\fix_taa.log 2>&1

echo [%date% %time%] child versions for %WORDS% >> logs\fix_taa.log
python scripts\childify.py --children-per-voice 4 --word-ids %WORDS% --workers 16 >> logs\fix_taa.log 2>&1

echo [%date% %time%] check + delete broken files >> logs\fix_taa.log
python scripts\check_child.py --fix --workers 8 >> logs\fix_taa.log 2>&1

echo [%date% %time%] re-make deleted files >> logs\fix_taa.log
python scripts\childify.py --children-per-voice 4 --workers 16 >> logs\fix_taa.log 2>&1

echo [%date% %time%] recheck re-made files >> logs\fix_taa.log
python scripts\check_child.py --recheck --workers 8 >> logs\fix_taa.log 2>&1

echo [%date% %time%] review pages >> logs\fix_taa.log
python scripts\make_review.py --set child --rows 75 --name review.html >> logs\fix_taa.log 2>&1
python scripts\make_review.py --set child --only distortion --rows 50 --name review_distortion.html >> logs\fix_taa.log 2>&1

echo [%date% %time%] PIPELINE DONE >> logs\fix_taa.log
