@echo off
setlocal

:: ========================================================
:: PIPELINE CONFIGURATION (Set your variables here ONCE)
:: ========================================================

:: CAE Settings
set CAE_NAME=texture_v2
set CAE_LATENT_DIM=64
set CAE_EPOCHS=15
set CAE_WEIGHTS_PATH=models\cae\%CAE_NAME%.pth

:: GNN & Image Settings
set IMG_SIZE=512
set SEGMENTS=300
set N_HOPS=1

:: Training Run Settings
set CHECKPOINT_DIR=checkpoints_512_full_run_%CAE_NAME%
set TRAIN_EPOCHS=100

echo ========================================================
echo STARTING WILDLIFE RE-ID PIPELINE
echo Target CAE: %CAE_NAME%
echo Target Checkpoints: %CHECKPOINT_DIR%
echo ========================================================
echo.

:: ========================================================
echo STEP 1: Training Texture Autoencoder (CAE)
:: ========================================================

if exist "%CAE_WEIGHTS_PATH%" (
    echo [SKIP] Found existing CAE weights at %CAE_WEIGHTS_PATH%. Skipping training.
) else (
    echo [TRAIN] No weights found. Initiating CAE Training...
    python .\src\ml_utils\gnn\cae.py ^
        --epochs %CAE_EPOCHS% ^
        --latent_dim %CAE_LATENT_DIM% ^
        --batch_size 128 ^
        --save_dir models\cae ^
        --model_name %CAE_NAME%

    if %errorlevel% neq 0 (
        echo [ERROR] CAE Training failed. Aborting pipeline.
        pause
        exit /b %errorlevel%
    )
)

echo.
:: ========================================================
echo STEP 2: Building LMDB Cache and Training GNN
:: ========================================================
python .\src\ml_utils\train_test_prototype.py ^
    --workers 6 ^
    --img_size %IMG_SIZE% ^
    --segments %SEGMENTS% ^
    --n_hops %N_HOPS% ^
    --epochs %TRAIN_EPOCHS% ^
    --data_mode lazy ^
    --checkpoint_dir %CHECKPOINT_DIR% ^
    --species hyena tiger nyala ^
    --features color pos hog cae ^
    --cae_weights_path %CAE_WEIGHTS_PATH% ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM%

if %errorlevel% neq 0 (
    echo [ERROR] GNN Training failed. Aborting pipeline.
    pause
    exit /b %errorlevel%
)

echo.
:: ========================================================
echo STEP 3: Running Comprehensive Evaluation
:: ========================================================
:: Note: Assuming you implemented the dictionary checkpoint saving, 
:: we no longer need to pass --n_hops to eval.py!

python .\src\ml_utils\eval.py ^
    --segments %SEGMENTS% ^
    --data_mode lazy ^
    --workers 4 ^
    --parallel_workers 14 ^
    --checkpoints_dir %CHECKPOINT_DIR% ^
    --batch_size 128 ^
    --features color pos hog cae ^
    --cae_weights_path %CAE_WEIGHTS_PATH% ^
    --cae_version %CAE_NAME% ^
    --cae_latent_dim %CAE_LATENT_DIM%

echo.
echo ========================================================
echo PIPELINE FULLY COMPLETE!
echo ========================================================
pause