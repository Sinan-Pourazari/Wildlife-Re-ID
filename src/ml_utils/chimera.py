import os
import subprocess
import glob
import sys

# ========================================================
# 1. STEP SELECTION: Toggle True/False to skip/run steps
# ========================================================
STEPS_TO_RUN = {
    "0.0_MERGE":         0,
    "0_SPLIT":            0,
    "2_TRAIN_CAE":        0,
    "3_TRAIN_GNN":        0,
    "4A_EVAL_KNOWN_GNN":  1,
    "4B_EVAL_KNOWN_WF":   1,
    "5A_EVAL_UNSEEN_GNN": 1,
    "5B_EVAL_UNSEEN_WF":  1,
    "6A_EVAL_MIXED_GNN":  1,
    "6B_EVAL_MIXED_WF":   1,
    "7A_EVAL_COMPETITION_GNN": 1,
    "7B_EVAL_COMPETITION_WF": 1,
}

# ========================================================
# 2. COLOR CONFIGURATION (Fox Style)
# ========================================================
FOX_ORANGE   = "\033[38;5;202m"
FOX_WHITE    = "\033[38;5;255m"
DIRT_BROWN   = "\033[38;5;94m"
DANGER_RED   = "\033[38;5;196m"
SUCCESS_LIME = "\033[38;5;118m"
RESET        = "\033[0m"

# ========================================================
# 3. PIPELINE CONFIGURATION
# ========================================================
COMMON_ROOT = "src/images"
REID_CSV = os.path.join(COMMON_ROOT, "reid-10k/metadata.csv")
CLEF_CSV = os.path.join(COMMON_ROOT, "animal-clef-2026/metadata.csv")
SHARED_CAE_DIR = "models/cae"
HOLDOUT_DATASET = "BalearicLizards"  # your original holdout (change to "NewtsKent" or "BalearicLizard" if needed)

# Hyperparameters
IMG_SIZE = 512
CAE_LATENT_DIM = 90
CAE_EPOCHS = 150
SEEDS_NUM_SP = 240
SEEDS_LEVELS = 4
NUM_HOG_BINS = 9
TRAIN_EPOCHS = 200
EDGE_STRATEGY = "spatial"
N_HOPS = 1

# Dataset Filter (for training)
DATASETS = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "AmvrakikosTurtles", "ATRW", "LeopardID2022"]

# Experiment Tracking
RUN_ID = "run_042_spatial"
EXPERIMENT_TAG = "animal_clef_2026_baseline"
CHECKPOINT_DIR = f"runs/{RUN_ID}_{EXPERIMENT_TAG}"

CAE_NAME = f"cae_dim{CAE_LATENT_DIM}_size{IMG_SIZE}_seeds{SEEDS_NUM_SP}_v2"
CAE_WEIGHTS_PATH = os.path.join(SHARED_CAE_DIR, f"{CAE_NAME}.pth")

# ========================================================
# 4. UTILITIES
# ========================================================
def run_cmd(step_name, cmd_list):
    """Executes command using the CURRENT python interpreter."""
    print(f"{FOX_ORANGE}--- EXECUTING: {step_name} ---{RESET}")
    try:
        if cmd_list[0] == "python":
            cmd_list[0] = sys.executable
        cmd = [str(c) for c in cmd_list if c]
        subprocess.run(cmd, check=True)
        print(f"{SUCCESS_LIME}[SUCCESS] {step_name} finished.{RESET}\n")
    except subprocess.CalledProcessError as e:
        print(f"{DANGER_RED}[ERROR] {step_name} failed. Return code: {e.returncode}.{RESET}")
        sys.exit(e.returncode)

def get_latest_checkpoint(directory):
    search_path = os.path.join(directory, "gnn", "*.pth")
    files = glob.glob(search_path)
    return max(files, key=os.path.getmtime) if files else None

# ========================================================
# 5. EXECUTION LOGIC
# ========================================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SHARED_CAE_DIR, exist_ok=True)
os.environ["HF_TOKEN"] = "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"

print(f"{FOX_ORANGE}========================================================{RESET}")
print(f"{FOX_ORANGE}STARTING WILDLIFE RE-ID PIPELINE {RESET}")
print(f"{DIRT_BROWN}Run Directory:{RESET} {FOX_WHITE}{CHECKPOINT_DIR}{RESET}")
print(f"{DIRT_BROWN}Target CAE:{RESET}    {FOX_WHITE}{CAE_WEIGHTS_PATH}{RESET}")
print(f"{FOX_ORANGE}========================================================{RESET}\n")

merged_csv = os.path.join(CHECKPOINT_DIR, "merged_metadata.csv")
train_csv  = os.path.join(CHECKPOINT_DIR, "train_split.csv")
test_csv   = os.path.join(CHECKPOINT_DIR, "test_split.csv")
competition_test_csv = os.path.join(CHECKPOINT_DIR, "competition_test.csv")

# ============================================================
# STEP 0.0 & STEP 0: MERGE & SPLIT (skip if splits already exist)
# ============================================================
need_merge = STEPS_TO_RUN.get("0.0_MERGE", False)
need_split = STEPS_TO_RUN.get("0_SPLIT", False)

if (need_merge or need_split) and (os.path.exists(train_csv) and os.path.exists(test_csv)):
    print(f"{FOX_WHITE}[SKIP] Split files already exist. Delete them to re-split.{RESET}\n")
    need_merge = False
    need_split = False

if need_merge:
    run_cmd("STEP 0.0: Merging Datasets",
            ["python", "src/ml_utils/merge_datasets.py",
             "--reid_csv", REID_CSV,
             "--clef_csv", CLEF_CSV,
             "--out_csv", merged_csv])

if need_split:
    split_cmd = ["python", "src/ml_utils/dataset_splitter.py",
                 "--csv_path", merged_csv, "--save_dir", CHECKPOINT_DIR]
    if DATASETS:
        split_cmd.extend(["--datasets"] + DATASETS)
    run_cmd("STEP 0: Initialize Dataset Split", split_cmd)

# STEP 2: CAE
if STEPS_TO_RUN["2_TRAIN_CAE"]:
    if os.path.exists(CAE_WEIGHTS_PATH):
        print(f"{FOX_WHITE}[SKIP] Found existing CAE weights at {CAE_WEIGHTS_PATH}.{RESET}\n")
    else:
        run_cmd("STEP 2: Training CAE",
                ["python", "src/ml_utils/gnn/cae.py", "--root_dir", COMMON_ROOT,
                 "--csv_path", train_csv, "--epochs", CAE_EPOCHS,
                 "--latent_dim", CAE_LATENT_DIM, "--save_dir", SHARED_CAE_DIR,
                 "--checkpoint_dir", CHECKPOINT_DIR,
                 "--model_name", CAE_NAME, "--img_size", IMG_SIZE])

# STEP 3: GNN TRAINING
if STEPS_TO_RUN["3_TRAIN_GNN"]:
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    resume_arg = ["--resume", latest_ckpt] if latest_ckpt else []
    if latest_ckpt:
        print(f"{SUCCESS_LIME}[INFO] Found checkpoint: {os.path.basename(latest_ckpt)}. Resuming...{RESET}")
    run_cmd("STEP 3: Training GNN",
            ["python", "src/ml_utils/train_test_prototype.py", "--root_dir", COMMON_ROOT,
             "--csv_path", train_csv, "--img_size", IMG_SIZE,
             "--seeds_num_superpixels", SEEDS_NUM_SP, "--seeds_num_levels", SEEDS_LEVELS,
             "--num_hog_bins", NUM_HOG_BINS, "--n_hops", N_HOPS,
             "--epochs", TRAIN_EPOCHS, "--checkpoint_dir", CHECKPOINT_DIR,
             "--features", "color", "pos", "hog", "shape", "lbp", "cae",
             "--cae_weights_path", CAE_WEIGHTS_PATH, "--cae_version", CAE_NAME,
             "--cae_latent_dim", CAE_LATENT_DIM,
             "--edge_strategy", EDGE_STRATEGY, "--margin_arc", 28.6] + resume_arg)

# EVALUATION SETTINGS
eval_args = ["--root_dir", COMMON_ROOT, "--img_size", IMG_SIZE,
             "--seeds_num_superpixels", SEEDS_NUM_SP,
             "--seeds_num_levels", SEEDS_LEVELS,
             "--num_hog_bins", NUM_HOG_BINS,
             "--checkpoints_dir", CHECKPOINT_DIR,
             "--features", "color", "pos", "hog", "shape", "lbp", "cae",
             "--cae_weights_path", CAE_WEIGHTS_PATH,
             "--cae_version", CAE_NAME, "--cae_latent_dim", CAE_LATENT_DIM,
             "--leiden_k1", 12, "--leiden_lambda", 0.2]

# 4A/4B – Known Domain
if STEPS_TO_RUN.get("4A_EVAL_KNOWN_GNN"):
    run_cmd("STEP 4A: Known Domain GNN", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", test_csv, "--batch_size", 512, "--parallel_workers", 14])
if STEPS_TO_RUN.get("4B_EVAL_KNOWN_WF"):
    run_cmd("STEP 4B: Known Domain WildFusion", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", test_csv, "--batch_size", 64, "--parallel_workers", 7, "--enable_wildfusion"])

# 5A/5B – Unseen Domain
if STEPS_TO_RUN.get("5A_EVAL_UNSEEN_GNN"):
    run_cmd("STEP 5A: Unseen Domain GNN", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", merged_csv, "--holdout_dataset", HOLDOUT_DATASET,
             "--batch_size", 512, "--parallel_workers", 14])
if STEPS_TO_RUN.get("5B_EVAL_UNSEEN_WF"):
    run_cmd("STEP 5B: Unseen Domain WildFusion", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", merged_csv, "--holdout_dataset", HOLDOUT_DATASET,
             "--batch_size", 64, "--parallel_workers", 7, "--enable_wildfusion"])

# 6A/6B – Mixed Domain
if STEPS_TO_RUN.get("6A_EVAL_MIXED_GNN"):
    run_cmd("STEP 6A: Mixed Domain GNN", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", merged_csv, "--base_test_csv", test_csv,
             "--holdout_dataset", HOLDOUT_DATASET,
             "--batch_size", 512, "--parallel_workers", 14])
if STEPS_TO_RUN.get("6B_EVAL_MIXED_WF"):
    run_cmd("STEP 6B: Mixed Domain WildFusion", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", merged_csv, "--base_test_csv", test_csv,
             "--holdout_dataset", HOLDOUT_DATASET,
             "--batch_size", 64, "--parallel_workers", 7, "--enable_wildfusion"])

# 7A/7B – Competition Test
if STEPS_TO_RUN.get("7A_EVAL_COMPETITION_GNN"):
    run_cmd("STEP 7A: Competition Test GNN", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", competition_test_csv, "--batch_size", 512, "--competition"])
if STEPS_TO_RUN.get("7B_EVAL_COMPETITION_WF"):
    run_cmd("STEP 7B: Competition Test WildFusion", ["python", "src/ml_utils/eval.py"] + eval_args +
            ["--csv_path", competition_test_csv, "--batch_size", 64,
             "--enable_wildfusion", "--competition"])

print(f"\n{SUCCESS_LIME}========================================================{RESET}")
print(f"{SUCCESS_LIME}PIPELINE FULLY COMPLETE!{RESET}")
print(f"{SUCCESS_LIME}========================================================{RESET}")