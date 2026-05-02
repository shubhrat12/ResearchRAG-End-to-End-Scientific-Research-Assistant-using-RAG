@echo off
echo ===================================================
echo Figure Detection Model Training Script (Improved)
echo ===================================================
echo.

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%PYTHONPATH%;.

REM Create data directories if they don't exist
if not exist "data\transfer_learning\prepared\figure_detection\train" mkdir "data\transfer_learning\prepared\figure_detection\train"
if not exist "data\transfer_learning\prepared\figure_detection\val" mkdir "data\transfer_learning\prepared\figure_detection\val"

REM Check if data exists
if not exist "data\transfer_learning\prepared\figure_detection\train\data.json" (
  echo Training data not found. Ensure data files are in place before running.
  exit /b 1
)

if not exist "data\transfer_learning\prepared\figure_detection\val\data.json" (
  echo Validation data not found. Ensure data files are in place before running.
  exit /b 1
)

echo Preparing figure detection data...
python create_figure_detection_data.py

echo.
echo Starting figure detection model training with 10 epochs and early stopping patience of 3...
python train_figure_model_simple_final.py --epochs 10 --batch-size 8 --max-length 256 --early-stopping-patience 3

echo.
echo Training completed! Check models/figure_detector/best_model for the best model by validation F1 score.
echo and models/figure_detector/final_model for the final trained model.
echo. 