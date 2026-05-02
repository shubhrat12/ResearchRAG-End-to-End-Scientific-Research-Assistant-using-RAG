@echo off
echo Training Reference Parser with improved entity detection and class imbalance handling

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Set environment variables for better CUDA error handling
set PYTHONPATH=%PYTHONPATH%;%CD%
set TORCH_USE_CUDA_DSA=1
set CUDA_LAUNCH_BLOCKING=1

REM Run data analysis first to understand the data
echo Running data analysis...
python analyze_reference_data.py

REM Train the model with our improvements (with CUDA safeguards)
echo Starting training on GPU with error safeguards...
python transfer_learning_fixed.py ^
  --model-type reference_parser ^
  --batch-size 4 ^
  --epochs 10 ^
  --learning-rate 1e-5 ^
  --unfreeze ^
  --weighted-loss ^
  --weight-decay 0.02 ^
  --early-stopping 3 ^
  --use-scheduler ^
  --gradient-clip 1.0 ^
  --label-smoothing 0.1 ^
  --max-length 256

if %errorlevel% neq 0 (
  echo Training encountered an error. Check logs for details.
) else (
  echo Training completed successfully!
)

echo Training complete! 