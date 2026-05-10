@echo off
setlocal EnableDelayedExpansion

:: ========================================================
:: ANIMALCLEF 2026 - FINAL SUBMISSION GENERATOR (HDBSCAN + WILDFUSION)
:: ========================================================

:: --- 1. MODEL SELECTION ---
:: Paste the exact path to your best GNN checkpoint here:
set "TARGET_PTH=runs\run_041_spatial_animal_clef_2026_baseline\gnn\gnn_reid_ep182_universal_20260430_173417.pth"

:: --- 2. ENSEMBLE CONFIGURATION ---
:: Set to 1 to use MegaDescriptor + GNN, set to 0 for GNN Only
set ENABLE_WILDFUSION=1

:: --- 3. HUGGING FACE AUTHENTICATION ---
:: Paste your HF token here to prevent download hangs (Required if WildFusion=1)
set HF_TOKEN=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

:: --- 4. CLUSTERING PARAMETERS (HDBSCAN) ---
:: k1 and lambda control the k-reciprocal re-ranking before HDBSCAN runs
set K1=12
set LAMBDA=0.2

:: --- 5. IMAGE & GRAPH PIPELINE (Must match your Training) ---
set RAW_DIR=src/images
set IMG_SIZE=512
set NUM_HOG_BINS=9
set CAE_LATENT_DIM=32
set CAE_WEIGHTS_PATH=models\cae\cae_dim32_size512_seeds240_v2.pth
set CAE_NAME=cae_dim32_size512_seeds240_v2

:: GRAPH STRUCTURE PARAMS 
set SEEDS_NUM_SUPERPIXELS=240
set SEEDS_NUM_LEVELS=4
set SEEDS_PRIOR=1
set SEEDS_HISTOGRAM_BINS=4
set N_HOPS=1

:: ========================================================
:: AUTOMATIC PATH RESOLUTION
:: ========================================================
if not exist "%TARGET_PTH%" (
    echo [DANGER] Model Checkpoint NOT FOUND at: %TARGET_PTH%
    pause
    exit /b 1
)

:: Extract root directory from checkpoint path
for %%I in ("%TARGET_PTH%") do set "GNN_DIR=%%~dpI"
set "GNN_DIR=!GNN_DIR:~0,-1!"
for %%I in ("!GNN_DIR!") do set "CHECKPOINT_DIR=%%~dpI"
set "CHECKPOINT_DIR=!CHECKPOINT_DIR:~0,-1!"

:: Locate the competition test CSV (usually in the run root)
set TEST_CSV=!CHECKPOINT_DIR!\competition_test.csv

:: Prepare the WildFusion Flag string
set WF_FLAG=
if "%ENABLE_WILDFUSION%"=="1" (
    set WF_FLAG=--enable_wildfusion
    echo [INFO] ENSEMBLING: MegaDescriptor-L-384 + GNN Texture Graph...
) else (
    echo [INFO] MODE: GNN Texture Graph Only...
)

:: ========================================================
:: EXECUTION
:: ========================================================
echo [INFO] Initializing Asynchronous Submission Pipeline...
echo [INFO] Model: %TARGET_PTH%
echo [INFO] Reranking: K1=%K1%, Lambda=%LAMBDA%

:: Exact parameter block requested + Submission overrides at the bottom
python .\src\ml_utils\eval.py ^
    --root_dir %RAW_DIR% ^
    --csv_path "%TEST_CSV%" ^
    --img_size %IMG_SIZE% ^
    --seeds_num_superpixels %SEEDS_NUM_SUPERPIXELS% ^
    --seeds_num_levels %SEEDS_NUM_LEVELS% ^
    --seeds_prior %SEEDS_PRIOR% ^
    --seeds_histogram_bins %SEEDS_HISTOGRAM_BINS% ^
    --num_hog_bins %NUM_HOG_BINS% ^
    --n_hops %N_HOPS% ^
    --data_mode auto ^
    --workers 0 ^
    --parallel_workers 0 ^
    --checkpoints_dir "%CHECKPOINT_DIR%" ^
    --batch_size 64 ^
    --features color pos hog shape lbp cae ^
    --cae_weights_path "%CAE_WEIGHTS_PATH%" ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM% ^
    --checkpoint_path "%TARGET_PTH%" ^
    --leiden_k1 %K1% ^
    --leiden_lambda %LAMBDA% ^
    --generate_submission ^
    %WF_FLAG%

echo ========================================================
echo SUBMISSION COMPLETE: Check "%CHECKPOINT_DIR%\submission.csv"
echo ========================================================
pause