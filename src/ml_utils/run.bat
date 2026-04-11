@echo off
setlocal

:: ========================================================
:: ENABLING ANSI COLORS
:: ========================================================
:: Generate the ESC character
for /F "delims=#" %%E in ('"prompt #$E# & echo on & for %%A in (1) do rem"') do set "ESC=%%E"

:: Define Fox Palette
set "FOX_ORANGE=%ESC%[38;5;202m"
set "FOX_WHITE=%ESC%[38;5;255m"
set "DIRT_BROWN=%ESC%[38;5;94m"
set "DANGER_RED=%ESC%[38;5;196m"
set "SUCCESS_LIME=%ESC%[38;5;118m"
set "RESET=%ESC%[0m"

:: ========================================================
:: PIPELINE CONFIGURATION
:: ========================================================
set CAE_NAME=texture_v8_16_deep_center_weighted_Felzenszwalb_v2
set CAE_LATENT_DIM=16
set CAE_EPOCHS=15
set CAE_WEIGHTS_PATH=models\cae\%CAE_NAME%.pth

set IMG_SIZE=512
set SEGMENTS=300
set N_HOPS=1
set EDGE_STRATEGY=attention
set CHECKPOINT_DIR=checkpoints_512_16_8_run_%EDGE_STRATEGY%_%CAE_NAME%
set TRAIN_EPOCHS=200

:: --- Optional: Resume Path ---
:: set RESUME_PATH=checkpoints_512_full_run_texture_v5_16_deep_center_weighted\gnn_reid_universal_20260410_100811.pth

echo %FOX_ORANGE%========================================================%RESET%
echo %FOX_ORANGE%STARTING WILDLIFE RE-ID PIPELINE%RESET%
echo %DIRT_BROWN%Target Checkpoints:%RESET% %FOX_WHITE%%CHECKPOINT_DIR%%RESET%
echo %DIRT_BROWN%Target CAE Weights:%RESET% %FOX_WHITE%%CAE_WEIGHTS_PATH%%RESET%
echo %FOX_ORANGE%========================================================%RESET%
echo.

:: ========================================================
echo %FOX_ORANGE%STEP 1: Training Texture Autoencoder (CAE)%RESET%
:: ========================================================
:: [cite: 1]
if exist "%CAE_WEIGHTS_PATH%" (
    echo %FOX_WHITE%[SKIP] Found existing CAE weights at %CAE_WEIGHTS_PATH%. Skipping training.%RESET%
) else (
    echo %FOX_ORANGE%[TRAIN] No weights found. Initiating CAE Training...%RESET%
    python .\src\ml_utils\gnn\cae.py ^
        --epochs %CAE_EPOCHS% ^
        --latent_dim %CAE_LATENT_DIM% ^
        --batch_size 128 ^
        --save_dir models\cae ^
        --model_name %CAE_NAME%

    if %errorlevel% neq 0 (
        echo %DANGER_RED%[ERROR] CAE Training failed. Aborting pipeline.%RESET%
        pause
        exit /b %errorlevel%
    )
)

echo.
:: ========================================================
echo %FOX_ORANGE%STEP 2: Building LMDB Cache and Training GNN%RESET%
:: ========================================================
:: [cite: 3]
echo %FOX_ORANGE%[TRAIN] Initiating GNN Training sequence...%RESET%
python .\src\ml_utils\train_test_prototype.py ^
    --workers 6 ^
    --img_size %IMG_SIZE% ^
    --segments %SEGMENTS% ^
    --n_hops %N_HOPS% ^
    --epochs %TRAIN_EPOCHS% ^
    --data_mode lazy ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --species hyena tiger nyala ^
    --features color pos hog cae shape lbp ^
    --cae_weights_path %CAE_WEIGHTS_PATH% ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --edge_strategy %EDGE_STRATEGY%

if %errorlevel% neq 0 (
    echo %DANGER_RED%[ERROR] GNN Training failed. Aborting pipeline.%RESET%
    pause
    exit /b %errorlevel%
)

echo.
:: ========================================================
echo %FOX_ORANGE%STEP 3: Running Comprehensive Evaluation%RESET%
:: ========================================================
:: [cite: 7]
echo %FOX_ORANGE%[EVAL] Running performance metrics...%RESET%
python .\src\ml_utils\eval.py ^
    --segments %SEGMENTS% ^
    --data_mode lazy ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 128 ^
    --features color pos hog cae shape lbp ^
    --cae_weights_path %CAE_WEIGHTS_PATH% ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM%

echo.
echo %SUCCESS_LIME%========================================================%RESET%
echo %SUCCESS_LIME%PIPELINE FULLY COMPLETE!%RESET%
echo %SUCCESS_LIME%========================================================%RESET%
pause