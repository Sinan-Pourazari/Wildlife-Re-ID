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
:: 1. Define all physical parameters first
set IMG_SIZE=512
set CAE_LATENT_DIM=32
set CAE_EPOCHS=15

set FELZ_SCALE=70.0
set FELZ_SIGMA=0.65
set FELZ_MIN_SIZE=150
set NUM_HOG_BINS=18
set N_HOPS=1
set EDGE_STRATEGY=hybrid

:: 2. Create safe strings for filenames (Replace '.' with 'p')
set SAFE_SCALE=%FELZ_SCALE:.=p%
set SAFE_SIGMA=%FELZ_SIGMA:.=p%

:: 3. DYNAMIC NAMING (Stitching the parameters together using the safe strings)
:: Result: cae_dim16_size512_scale70p0_sigma0p65
set CAE_NAME=cae_dim%CAE_LATENT_DIM%_size%IMG_SIZE%_scale%SAFE_SCALE%_sigma%SAFE_SIGMA%

:: 4. Assign Paths
set CAE_WEIGHTS_PATH=models\cae\%CAE_NAME%.pth
set CHECKPOINT_DIR=checkpoints_%IMG_SIZE%_scale%SAFE_SCALE%_sigma%SAFE_SIGMA%_hops%N_HOPS%_%EDGE_STRATEGY%_dim%CAE_LATENT_DIM%_v8
::set CHECKPOINT_DIR=debug
set TRAIN_EPOCHS=150

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
if exist "%CAE_WEIGHTS_PATH%" (
    echo %FOX_WHITE%[SKIP] Found existing CAE weights at %CAE_WEIGHTS_PATH%. Skipping training.%RESET%
) else (
    echo %FOX_ORANGE%[TRAIN] No weights found. Initiating CAE Training...%RESET%
    python .\src\ml_utils\gnn\cae.py ^
        --epochs %CAE_EPOCHS% ^
        --latent_dim %CAE_LATENT_DIM% ^
        --batch_size 128 ^
        --save_dir models\cae ^
        --model_name %CAE_NAME% ^
        --img_size %IMG_SIZE%

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
echo %FOX_ORANGE%[TRAIN] Initiating GNN Training sequence...%RESET%
python .\src\ml_utils\train_test_prototype.py ^
    --workers 6 ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --n_hops %N_HOPS% ^
    --epochs %TRAIN_EPOCHS% ^
    --data_mode lazy ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --species tiger hyena nyala ^
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
echo %FOX_ORANGE%[EVAL] Running performance metrics...%RESET%
python .\src\ml_utils\eval.py ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 256 ^
    --features color pos hog cae shape lbp ^
    --cae_weights_path %CAE_WEIGHTS_PATH% ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM%

echo.
echo %SUCCESS_LIME%========================================================%RESET%
echo %SUCCESS_LIME%PIPELINE FULLY COMPLETE!%RESET%
echo %SUCCESS_LIME%========================================================%RESET%
pause