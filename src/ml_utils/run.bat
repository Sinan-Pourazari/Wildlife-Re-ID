@echo off
echo ========================================================
echo   Starting Wildlife Re-ID Pipeline
echo ========================================================
echo.

:: --- STEP 1: TRAINING ---
echo [1/2] Launching Training Script...
:: Put your exact training command here. I removed the subset_fraction!
python .\src\ml_utils\train_test_prototype.py --workers 6 --img_size 1024 --segments 300 --epochs 200 --data_mode lazy  --checkpoint_dir checkpoints_long_run_v6_ext_feat --rebuild

:: Check if the training crashed. If it did, stop the script.
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training crashed or was interrupted! Stopping pipeline.
    pause
    exit /b %errorlevel%
)
echo.
echo [SUCCESS] Training completed successfully!
echo.


:: --- STEP 2: EVALUATION ---
echo [2/2] Launching Evaluation Script...
:: Put your exact evaluation command here
python .\src\ml_utils\eval.py  --segments 300 --data_mode lazy --workers 2  --checkpoints_dir checkpoints_long_run_v6_ext_feat

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Evaluation crashed!
    pause
    exit /b %errorlevel%
)
echo.
echo ========================================================
echo   Pipeline Finished! Check your HTML and CSV files.
echo ========================================================
pause