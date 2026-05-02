@echo off
echo ===================================================
echo Section Classifier Training Script
echo ===================================================
echo.

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%PYTHONPATH%;.

REM Ensure logs directory exists
if not exist "logs" mkdir logs

REM Create model directory if it doesn't exist
if not exist "models\section_classifier" mkdir "models\section_classifier"

echo Running section classifier training...
echo ---------------------------------------------------
python transfer_learning_fixed.py --model-type section_classifier --batch-size 8 --epochs 10 --unfreeze

echo.
echo ===================================================
echo Section classifier training completed!
echo =================================================== 