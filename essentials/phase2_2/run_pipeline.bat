@echo off
REM Scientific NLP Pipeline (Phase 2.2) Runner
REM This script runs the complete Phase 2.2 pipeline

echo ===================================================
echo Scientific NLP Pipeline (Phase 2.2)
echo ===================================================
echo.

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Ensure output directory exists
if not exist data\derived\phase2.2_output mkdir data\derived\phase2.2_output

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%PYTHONPATH%;.

REM Run the pipeline with test mode (small sample) first
echo Running pipeline in test mode (10 documents)...
python run_phase2_2_pipeline.py --max-docs 10 --batch-size 5

REM If test was successful, offer to run full pipeline
echo.
echo Test run completed. Check the output files in data\derived\phase2.2_output.
echo.
set /p RUN_FULL=Do you want to run the full pipeline on all documents? (y/n): 

if /i "%RUN_FULL%"=="y" (
    echo.
    echo Running full pipeline on all documents...
    python run_phase2_2_pipeline.py --batch-size 50
) else (
    echo.
    echo Full pipeline run skipped.
)

echo.
echo ===================================================
echo Pipeline execution completed!
echo Check output files in data\derived\phase2.2_output
echo =================================================== 