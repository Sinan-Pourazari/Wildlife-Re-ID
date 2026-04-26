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

:: set the root to the shared parent folder!
set COMMON_ROOT=src/images

:: Point to both individual CSVs
set REID_CSV=src/images/reid-10k/metadata.csv
set CLEF_CSV=src/images/animal-clef-2026/metadata.csv

:: --- DATASET SELECTION ---
:: Uncomment the dataset you want to use (and comment out the other):

:: Option A: Wildlife ReID-10k Dataset (Active)
set RAW_DIR=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\reid-10k
set CSV_PATH=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\reid-10k\metadata.csv

:: Option B: AnimalCLEF 2026 Dataset
:: set RAW_DIR=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026
:: set CSV_PATH=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\metadata.csv
:: -------------------------

:: [Persistent directory for CAE models so they aren't lost in timestamped folders
set SHARED_CAE_DIR=models\cae

set IMG_SIZE=256
set CAE_LATENT_DIM=32
set CAE_EPOCHS=150
set FELZ_SCALE=70.0
set FELZ_SIGMA=0.65
set FELZ_MIN_SIZE=300
set NUM_HOG_BINS=9
set N_HOPS=1
set EDGE_STRATEGY=hybrid
set TRAIN_EPOCHS=200
:: Define your target species here! (Leave blank to use the whole dataset)
:: E.g., set DATASETS=tiger turtle leopard
set DATASETS=LynxID2025 SalamanderID2025 SeaTurtleID2022 AmvrakikosTurtles ATRW LeopardID2022 SeaStarReID2023

:: ========================================================
:: NEW DYNAMIC CHECKPOINT NAMING SYSTEM
:: ========================================================
:: Manually update this ID for each new experiment
set RUN_ID=run_021

set EXPERIMENT_TAG=animal_clef_2026_baseline
set CHECKPOINT_DIR=runs\!RUN_ID!_!EXPERIMENT_TAG!

:: Safe formatting for CAE names
set SAFE_SCALE=%FELZ_SCALE:.=p%
set SAFE_SIGMA=%FELZ_SIGMA:.=p%
set CAE_NAME=cae_dim%CAE_LATENT_DIM%_size%IMG_SIZE%_scale%SAFE_SCALE%_sigma%SAFE_SIGMA%_clef_big_v2

:: Point the weights path to the persistent shared directory
set CAE_WEIGHTS_PATH=%SHARED_CAE_DIR%\%CAE_NAME%.pth

:: Define CSV paths inside the new directory
set PROCESSED_CSV=!CHECKPOINT_DIR!\pipeline_metadata.csv
set TRAIN_CSV=!CHECKPOINT_DIR!\train_split.csv
set TEST_CSV=!CHECKPOINT_DIR!\test_split.csv
set GOLBAL_CSV=!CHECKPOINT_DIR!\merged_metadata.csv
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
echo %FOX_ORANGE%STEP 0.0: Merging Datasets%RESET%
:: ========================================================
set MERGED_CSV=!CHECKPOINT_DIR!\merged_metadata.csv

python .\src\ml_utils\merge_datasets.py ^
    --reid_csv %REID_CSV% ^
    --clef_csv %CLEF_CSV% ^
    --out_csv !MERGED_CSV!

:: Now we tell the rest of the pipeline to use the shared root and the merged CSV
set RAW_DIR=%COMMON_ROOT%
set CSV_PATH=!MERGED_CSV!

echo.
:: ========================================================
echo %FOX_ORANGE%STEP 0: Initialize Dataset Split %RESET%
:: ========================================================

:: Build the base arguments
set "SPLITTER_ARGS=--csv_path "%CSV_PATH%" --save_dir "%CHECKPOINT_DIR%""

:: Safely append the dataset filter if DATASETS is not empty
if not "%DATASETS%"=="" (
    set "SPLITTER_ARGS=!SPLITTER_ARGS! --datasets %DATASETS%"
)

python .\src\ml_utils\dataset_splitter.py !SPLITTER_ARGS!

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

:: ---  RESUME LOGIC ---
set "RESUME_ARG="
set "LATEST_CHECKPOINT="
if exist "!CHECKPOINT_DIR!\gnn\*.pth" (
    :: Sorts files by date (/o-d) and grabs the first one it sees (the newest)
    for /f "delims=" %%I in ('dir "!CHECKPOINT_DIR!\gnn\*.pth" /b /o-d 2^>nul') do (
        set "LATEST_CHECKPOINT=%%I"
        goto :found_ckpt
    )
)
:found_ckpt
if defined LATEST_CHECKPOINT (
    echo %SUCCESS_LIME%[INFO] Found existing checkpoint: !LATEST_CHECKPOINT!. Resuming training...%RESET%
    set "RESUME_ARG=--resume "!CHECKPOINT_DIR!\gnn\!LATEST_CHECKPOINT!""
) else (
    echo %FOX_WHITE%[INFO] No existing checkpoints found. Starting GNN training from scratch...%RESET%
)

echo %FOX_ORANGE%[TRAIN] Initiating GNN Training sequence...%RESET%
python .\src\ml_utils\train_test_prototype.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %TRAIN_CSV% ^
    --workers 8 ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --n_hops %N_HOPS% ^
    --epochs %TRAIN_EPOCHS% ^
    --data_mode auto ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --edge_strategy %EDGE_STRATEGY% ^
    --margin_arc 28.6 ^
    !RESUME_ARG!

if !errorlevel! neq 0 (
    echo %DANGER_RED%[ERROR] GNN Training failed. Aborting pipeline.%RESET%
    pause
    exit /b !errorlevel!
)
echo.
::--csv_path %TEST_CSV% ^
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
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 1024 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% 
    ::--holdout_dataset HyenaID2022

:: ========================================================
echo %FOX_ORANGE%STEP 5: Running Comprehensive Evaluation on Holdout Domain%RESET%
:: ========================================================
echo %FOX_ORANGE%[EVAL] Running performance metrics...%RESET%
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %GOLBAL_CSV% ^
    --img_size %IMG_SIZE% ^
    --felz_scale %FELZ_SCALE% ^
    --felz_sigma %FELZ_SIGMA% ^
    --felz_min_size %FELZ_MIN_SIZE% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 1024 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --holdout_dataset HyenaID2022
echo %SUCCESS_LIME%========================================================%RESET%
echo %SUCCESS_LIME%PIPELINE FULLY COMPLETE!%RESET%
echo %SUCCESS_LIME%========================================================%RESET%
pause