@echo off
:: CRITICAL: This enables the script to catch Python crashes properly
setlocal EnableDelayedExpansion

:: ========================================================
:: ENABLING ANSI COLORS
:: ========================================================
for /F "delims=#" %%E in ('"prompt #$E# & echo on & for %%A in (1) do rem"') do set "ESC=%%E"
set "FOX_ORANGE=%ESC%[38;5;202m"
set "FOX_WHITE=%ESC%[38;5;255m"
set "DIRT_BROWN=%ESC%[38;5;94m"
set "DANGER_RED=%ESC%[38;5;196m"
set "SUCCESS_LIME=%ESC%[38;5;118m"
set "RESET=%ESC%[0m"

:: ========================================================
:: PIPELINE CONFIGURATION
:: ========================================================
set RAW_DIR=src/images/animal-clef-2026
set CSV_PATH=src/images/animal-clef-2026/metadata.csv

:: [NEW] Persistent directory for CAE models so they aren't lost in timestamped folders
set SHARED_CAE_DIR=models\cae

set IMG_SIZE=256
set CAE_LATENT_DIM=24
set CAE_EPOCHS=600
set FELZ_SCALE=70.0
set FELZ_SIGMA=0.65
set FELZ_MIN_SIZE=300
set NUM_HOG_BINS=9
set N_HOPS=1
set EDGE_STRATEGY=hybrid
set DATASETS=
set TRAIN_EPOCHS=1000

:: ========================================================
:: NEW DYNAMIC CHECKPOINT NAMING SYSTEM
:: ========================================================
:: 1. Get a reliable, locale-independent timestamp (YYYYMMDD_HHMMSS)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set RUN_TIMESTAMP=!datetime:~0,8!_!datetime:~8,6!

set EXPERIMENT_TAG=animal_clef_2026_baseline
set CHECKPOINT_DIR=runs\!RUN_TIMESTAMP!_!EXPERIMENT_TAG!

:: Safe formatting for CAE names
set SAFE_SCALE=%FELZ_SCALE:.=p%
set SAFE_SIGMA=%FELZ_SIGMA:.=p%
set CAE_NAME=cae_dim%CAE_LATENT_DIM%_size%IMG_SIZE%_scale%SAFE_SCALE%_sigma%SAFE_SIGMA%_clef

:: [NEW] Point the weights path to the persistent shared directory
set CAE_WEIGHTS_PATH=%SHARED_CAE_DIR%\%CAE_NAME%.pth

:: Define CSV paths inside the new directory
set PROCESSED_CSV=!CHECKPOINT_DIR!\pipeline_metadata.csv
set TRAIN_CSV=!CHECKPOINT_DIR!\train_split.csv
set TEST_CSV=!CHECKPOINT_DIR!\test_split.csv

:: ========================================================
:: SAVE HYPERPARAMETERS TO LOG
:: ========================================================
:: Create the folders immediately
mkdir "!CHECKPOINT_DIR!" 2>nul
mkdir "!SHARED_CAE_DIR!" 2>nul

echo =================================== > "!CHECKPOINT_DIR!\run_config.log"
echo EXPERIMENT: !EXPERIMENT_TAG! >> "!CHECKPOINT_DIR!\run_config.log"
echo TIMESTAMP: !RUN_TIMESTAMP! >> "!CHECKPOINT_DIR!\run_config.log"
echo SHARED_CAE: !CAE_WEIGHTS_PATH! >> "!CHECKPOINT_DIR!\run_config.log"
echo =================================== >> "!CHECKPOINT_DIR!\run_config.log"
echo IMG_SIZE: %IMG_SIZE% >> "!CHECKPOINT_DIR!\run_config.log"
echo CAE_LATENT_DIM: %CAE_LATENT_DIM% >> "!CHECKPOINT_DIR!\run_config.log"
echo EDGE_STRATEGY: %EDGE_STRATEGY% >> "!CHECKPOINT_DIR!\run_config.log"
echo N_HOPS: %N_HOPS% >> "!CHECKPOINT_DIR!\run_config.log"
echo FELZ_SCALE: %FELZ_SCALE% >> "!CHECKPOINT_DIR!\run_config.log"
echo FELZ_SIGMA: %FELZ_SIGMA% >> "!CHECKPOINT_DIR!\run_config.log"
echo FELZ_MIN_SIZE: %FELZ_MIN_SIZE% >> "!CHECKPOINT_DIR!\run_config.log"
echo SPECIES: %SPECIES% >> "!CHECKPOINT_DIR!\run_config.log"

echo %FOX_ORANGE%========================================================%RESET%
echo %FOX_ORANGE%STARTING WILDLIFE RE-ID PIPELINE%RESET%
echo %DIRT_BROWN%Run Directory:%RESET% %FOX_WHITE%!CHECKPOINT_DIR!%RESET%
echo %DIRT_BROWN%Target CAE:%RESET% %FOX_WHITE%!CAE_WEIGHTS_PATH!%RESET%
echo %FOX_ORANGE%========================================================%RESET%
echo.

:: ========================================================
echo %FOX_ORANGE%STEP 0: Initialize Dataset Split %RESET%
:: ========================================================
python .\src\ml_utils\dataset_splitter.py ^
    --csv_path %CSV_PATH% ^
    --save_dir %CHECKPOINT_DIR% 
if !errorlevel! neq 0 (
    echo %DANGER_RED%[ERROR] Dataset splitting failed. Aborting.%RESET%
    pause
    exit /b !errorlevel!
)

:: ========================================================
echo %FOX_ORANGE%STEP 2: Training Texture Autoencoder (CAE)%RESET%
:: ========================================================
:: This will now successfully find the CAE if it was trained in a previous run!
if exist "%CAE_WEIGHTS_PATH%" (
    echo %FOX_WHITE%[SKIP] Found existing CAE weights at %CAE_WEIGHTS_PATH%. Skipping training.%RESET%
) else (
    echo %FOX_ORANGE%[TRAIN] No weights found. Initiating CAE Training...%RESET%
    python .\src\ml_utils\gnn\cae.py ^
        --root_dir %RAW_DIR% ^
        --csv_path %TRAIN_CSV% ^
        --epochs %CAE_EPOCHS% ^
        --latent_dim %CAE_LATENT_DIM% ^
        --save_dir %SHARED_CAE_DIR% ^
        --checkpoint_dir %CHECKPOINT_DIR% ^
        --model_name %CAE_NAME% ^
        --img_size %IMG_SIZE%
        
    if !errorlevel! neq 0 (
        echo %DANGER_RED%[ERROR] CAE Training failed. Aborting pipeline.%RESET%
        pause
        exit /b !errorlevel!
    )
)
echo.

:: ========================================================
echo %FOX_ORANGE%STEP 3: Building LMDB Cache and Training GNN%RESET%
:: ========================================================
echo %FOX_ORANGE%[TRAIN] Initiating GNN Training sequence...%RESET%
python .\src\ml_utils\train_test_prototype.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %TRAIN_CSV% ^
    --workers 0 ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --n_hops %N_HOPS% ^
    --epochs %TRAIN_EPOCHS% ^
    --data_mode memory ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --edge_strategy %EDGE_STRATEGY% 

if !errorlevel! neq 0 (
    echo %DANGER_RED%[ERROR] GNN Training failed. Aborting pipeline.%RESET%
    pause
    exit /b !errorlevel!
)
echo.

:: ========================================================
echo %FOX_ORANGE%STEP 4: Running Comprehensive Evaluation%RESET%
:: ========================================================
echo %FOX_ORANGE%[EVAL] Running performance metrics...%RESET%
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %TEST_CSV% ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 16 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 1024 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% 
echo.

echo %SUCCESS_LIME%========================================================%RESET%
echo %SUCCESS_LIME%PIPELINE FULLY COMPLETE!%RESET%
echo %SUCCESS_LIME%========================================================%RESET%
pause