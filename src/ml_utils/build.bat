@echo off
title Overnight Re-ID Training
echo ========================================================
echo Starting Overnight Re-ID Training Pipeline
echo ========================================================

:: --- COMMON SETTINGS ---
set WORKERS=6
set EPOCHS=1
set DATA_MODE=lazy
set N_HOPS=2

:: Optional: If you need to activate your virtual environment first, uncomment the line below:
:: call .venv_wildlife\Scripts\activate.bat

:: --- SWEEP PARAMETERS ---
:: Add or remove numbers here separated by spaces
set IMG_SIZES=512 256
set SEGMENTS_LIST=100 200 300 400 500

echo Starting parameter sweep...
echo.

:: Loop through every Image Size
for %%I in (%IMG_SIZES%) do (
    
    :: Inside that, loop through every Segment amount
    for %%S in (%SEGMENTS_LIST%) do (
        
        echo ========================================================
        echo Starting Run: Image Size: %%I ^| Segments: %%S ^| Hops: %N_HOPS%
        echo ========================================================
        
        python .\src\ml_utils\train_test_prototype.py ^
            --workers %WORKERS% ^
            --img_size %%I ^
            --epochs %EPOCHS% ^
            --data_mode %DATA_MODE% ^
            --segments %%S ^
            --n_hops %N_HOPS% ^
            --checkpoint_dir checkpoints_%%I_seg%%S_hop%N_HOPS%_v1
            
        echo.
    )
)

echo ========================================================
echo ALL OVERNIGHT TRAINING JOBS COMPLETED!
echo ========================================================
pause