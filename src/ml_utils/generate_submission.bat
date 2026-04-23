@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo ANIMALCLEF SUBMISSION GENERATOR
echo ========================================================

:: ========================================================
:: 1. PASTE THE EXACT PATH TO YOUR MODEL HERE:
:: ========================================================
set "TARGET_PTH=runs\run_008_animal_clef_2026_baseline\gnn\gnn_reid_ep588_universal_20260423_154456.pth"
if not exist "%TARGET_PTH%" (
    echo [ERROR] Could not find checkpoint: %TARGET_PTH%
    pause
    exit /b 1
)

:: Automatically extract the root checkpoint directory from the model path
for %%I in ("%TARGET_PTH%") do set "GNN_DIR=%%~dpI"
set "GNN_DIR=!GNN_DIR:~0,-1!"
for %%I in ("!GNN_DIR!") do set "CHECKPOINT_DIR=%%~dpI"
set "CHECKPOINT_DIR=!CHECKPOINT_DIR:~0,-1!"

:: Recreate standard pipeline vars
set RAW_DIR=src/images/animal-clef-2026
set TEST_CSV=!CHECKPOINT_DIR!\competition_test.csv
set IMG_SIZE=256
set FELZ_SCALE=70.0
set FELZ_SIGMA=0.65
set FELZ_MIN_SIZE=300
set NUM_HOG_BINS=9
set CAE_LATENT_DIM=24
set CAE_WEIGHTS_PATH=models\cae\cae_dim24_size256_scale70p0_sigma0p65_clef.pth
set CAE_NAME=cae_dim24_size256_scale70p0_sigma0p65_clef

:: ========================================================
:: 2. LEIDEN CLUSTERING PARAMS
:: ========================================================
set LEIDEN_THRESH=0.36
set LEIDEN_K1=20
set LEIDEN_LAMBDA=0.4
echo.
echo [INFO] Generating submission.csv using %TARGET_PTH%...
echo [INFO] Using Params: Thresh=%LEIDEN_THRESH%, K1=%LEIDEN_K1%, Lambda=%LEIDEN_LAMBDA%
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path "%TEST_CSV%" ^
    --checkpoints_dir "%CHECKPOINT_DIR%" ^
    --checkpoint_path "%TARGET_PTH%" ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --batch_size 1024 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --leiden_thresh %LEIDEN_THRESH% ^
    --leiden_k1 %LEIDEN_K1% ^
    --leiden_lambda %LEIDEN_LAMBDA% ^
    --generate_submission

echo.
pause