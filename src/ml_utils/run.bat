@echo off
set HF_TOKEN=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
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
set RAW_DIR=%COMMON_ROOT%
set CSV_PATH=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\reid-10k\metadata.csv

:: Option B: AnimalCLEF 2026 Dataset
:: set RAW_DIR=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026
:: set CSV_PATH=C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\metadata.csv
:: -------------------------

:: [Persistent directory for CAE models so they aren't lost in timestamped folders
set SHARED_CAE_DIR=models\cae

set IMG_SIZE=512
set CAE_LATENT_DIM=90
set CAE_EPOCHS=150
set SEEDS_NUM_SUPERPIXELS=240
set SEEDS_NUM_LEVELS=4
set SEEDS_PRIOR=1
set SEEDS_HISTOGRAM_BINS=4
set NUM_HOG_BINS=9
set N_HOPS=1
set EDGE_STRATEGY=spatial
set TRAIN_EPOCHS=210
:: Define target DATASETS here! (Leave blank to use the whole dataset)
:: E.g., set DATASETS=tiger turtle leopard
set DATASETS=LynxID2025 SalamanderID2025 SeaTurtleID2022 AmvrakikosTurtles ATRW LeopardID2022

:: ========================================================
:: NEW DYNAMIC CHECKPOINT NAMING SYSTEM
:: ========================================================
:: Manually update this ID for each new experiment
set RUN_ID=run_037_spatial

set EXPERIMENT_TAG=animal_clef_2026_baseline
set CHECKPOINT_DIR=runs\!RUN_ID!_!EXPERIMENT_TAG!

:: Safe formatting for CAE names
set SAFE_SCALE=%FELZ_SCALE:.=p%
set SAFE_SIGMA=%FELZ_SIGMA:.=p%
set CAE_NAME=cae_dim%CAE_LATENT_DIM%_size%IMG_SIZE%_seeds%SEEDS_NUM_SUPERPIXELS%_v2

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
echo %FOX_ORANGE%STEP 4A: KNOWN DOMAIN Evaluation (GNN Only)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %TEST_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 512 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM%

:: ========================================================
echo %FOX_ORANGE%STEP 4B: KNOWN DOMAIN Evaluation (WildFusion)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %TEST_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 7 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 64 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --enable_wildfusion

:: ========================================================
echo %FOX_ORANGE%STEP 5A: UNKNOWN DOMAIN Evaluation (GNN Only)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %GOLBAL_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 512 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --holdout_dataset HyenaID2022

:: ========================================================
echo %FOX_ORANGE%STEP 5B: UNKNOWN DOMAIN Evaluation (WildFusion)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %GOLBAL_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 7 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 64 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --holdout_dataset HyenaID2022 ^
    --enable_wildfusion

:: ========================================================
echo %FOX_ORANGE%STEP 6A: MIXED DOMAIN Evaluation (GNN Only)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %GOLBAL_CSV% ^
    --base_test_csv %TEST_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 512 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --holdout_dataset HyenaID2022

:: ========================================================
echo %FOX_ORANGE%STEP 6B: MIXED DOMAIN Evaluation (WildFusion)%RESET%
:: ========================================================
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path %GOLBAL_CSV% ^
    --base_test_csv %TEST_CSV% ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --data_mode auto ^
    --workers 4 ^
    --parallel_workers 7 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 64 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --holdout_dataset HyenaID2022 ^
    --enable_wildfusion

echo %SUCCESS_LIME%========================================================%RESET%
echo %SUCCESS_LIME%PIPELINE FULLY COMPLETE!%RESET%
echo %SUCCESS_LIME%========================================================%RESET%
pause