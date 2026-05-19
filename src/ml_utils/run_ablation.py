import os
import subprocess
import glob
import sys
import time
import traceback
import json
import pandas as pd
# ========================================================
# 1. STEP SELECTION
# ========================================================
STEPS_TO_RUN = {
    "0.0_MERGE":          1,
    "0_SPLIT":            1,
    "2_TRAIN_CAE":        1,
    "3_TRAIN_GNN":        1,
    
    # --- HARDCODED HDBSCAN EVALUATIONS ---
    "4A_EVAL_KNOWN_GNN":  1,
    "4B_EVAL_KNOWN_WF":   0,
    "4C_EVAL_KNOWN_PER_SPECIES": 1,
    "5A_EVAL_UNSEEN_GNN": 1,
    "5B_EVAL_UNSEEN_WF":  0,
    "6A_EVAL_MIXED_GNN":  1,
    "6B_EVAL_MIXED_WF":   0,
    "7A_EVAL_COMPETITION_GNN": 0,
    "7B_EVAL_COMPETITION_WF": 0,
    "8_CROSS_EVAL_MATRIX": 1,
    
    # --- OPTIMIZED HDBSCAN EVALUATIONS ---
    "4A_EVAL_KNOWN_GNN_OPT": 1,
    "4C_EVAL_KNOWN_PER_SPECIES_OPT": 1,
    "5A_EVAL_UNSEEN_GNN_OPT": 1,
    "6A_EVAL_MIXED_GNN_OPT": 1,
    "8_CROSS_EVAL_MATRIX_OPT": 1,

    # --- POST-PROCESSING ABLATION ---
    "4A_EVAL_KNOWN_NO_RERANK": 1,
    "4C_EVAL_KNOWN_PER_SPECIES_NO_RERANK": 1, 
    "5A_EVAL_UNSEEN_GNN_NO_RERANK": 1, 
    "4A_EVAL_KNOWN_OPT_NO_RERANK": 1,
    "6A_EVAL_MIXED_GNN_NO_RERANK" : 1,
    # --- VISUALIZATIONS ---
    "9_GENERATE_TSNE": 0, 
}

# ========================================================
# 2. COLOR CONFIGURATION
# ========================================================
FOX_ORANGE   = "\033[38;5;202m"
FOX_WHITE    = "\033[38;5;255m"
DIRT_BROWN   = "\033[38;5;94m"
DANGER_RED   = "\033[38;5;196m"
SUCCESS_LIME = "\033[38;5;118m"
RESET        = "\033[0m"

# ========================================================
# 3. GLOBAL PATHS & SETTINGS
# ========================================================
COMMON_ROOT = "src/images"
REID_CSV = os.path.join(COMMON_ROOT, "reid-10k/metadata.csv")
CLEF_CSV = os.path.join(COMMON_ROOT, "animal-clef-2026/metadata.csv")
SHARED_CAE_DIR = "models/cae"
HOLDOUT_DATASET = "HyenaID2022" 

# The 3 core AnimalCLEF 2026 species (dataset name → short tag used for step keys / eval_suffix)
CLEF_SPECIES = [
    ("LynxID2025",       "lynx"),
    ("SalamanderID2025", "salamander"),
    ("SeaTurtleID2022",  "sea_turtle"),
]

# ========================================================
# 4. ABLATION CONFIGURATIONS
# ========================================================
# BASE_CONFIG acts as the strict scientific baseline (Core Dataset + Clean Features)
BASE_CONFIG = {
    "seed": 42, # Reproducibility enforcement
    "epochs": 200,
    "batch_size": 512,
    "img_size": 512,
    "margin_arc": 28.6,
    "seeds_num_superpixels": 240,
    "seeds_num_levels": 4,
    "num_hog_bins": 9,
    "n_hops": 1,
    "edge_strategy": "spatial",
    "cae_latent_dim": 90,
    "pooling_type": "gem",
    "ortho_weight": 1.0,
    "emb_dim": 512,
    "features": ["color", "pos", "cae"], # The pure topological baseline
    "split_datasets": ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"] # Core AnimalCLEF baseline
}

ABLATIONS = [

    #--- EXPLORATION 0: The Control Group ---
    {"name": "Exp0_R00_Baseline", "args": { "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention"}},

    # --- EXPLORATION 1: Graph Construction (Edges & Receptive Field) ---
    {"name": "Exp1_R01_Edges_Attn", "args": {"edge_strategy": "attention", "k_neighbors": 5}},
    {"name": "Exp1_R02_Edges_Hybrid", "args": {"edge_strategy": "hybrid", "k_neighbors": 5}},
    # NEW: Test how latent edges survive deeper message passing
    {"name": "Exp1_R02b_Edges_Hybrid_2L", "args": {"edge_strategy": "hybrid", "k_neighbors": 5, "gnn_layers": 2, "use_jk": True, "jk_mode": "attention"}},
    #{"name": "Exp1_R03_Hops_2", "args": {"n_hops": 2}}, # FLAWED WILL RESULT IN OOM

    # --- EXPLORATION 2: Depth & Routing ---
    {"name": "Exp2_R04_GCN_1L", "args": {"gnn_layers": 1, "gnn_type": "gcn"}},
    # NEW: Bridge the gap to see if standard GCN dies at 2 layers without attention
    {"name": "Exp2_R04b_GCN_2L", "args": {"gnn_layers": 2, "gnn_type": "gcn"}},
    #{"name": "Exp2_R05_GATv2_3L_JK_Attn", "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention"}}, # Redundant is baseline
    {"name": "Exp2_R06_GATv2_3L_JK_Mean", "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "mean"}},
    {"name": "Exp2_R06b_GATv2_3L_No_JK", "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": False}},
    {"name": "Exp2_R07_GATv2_1L", "args": {"gnn_layers": 1, "gnn_type": "gatv2"}}, # Expected to be the best router
    # NEW: Find the exact sweet spot between the 1L probe and the 3L baseline
    {"name": "Exp2_R07b_GATv2_2L_JK_Attn", "args": {"gnn_layers": 2, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention"}},

    # --- EXPLORATION 3: Feature Vocabulary ---
    {"name": "Exp3_R08_Color", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["color"]}},
    {"name": "Exp3_R09_ClassicVision", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["color", "hog", "pos"]}},
    {"name": "Exp3_R10_PureTexture", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae"]}},
    {"name": "Exp3_R10A_Texture_Pos", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae", "pos"]}},
    
    # 2. Your actual intended Baseline. (The sweet spot of deep texture + geometry + color).
    {"name": "Exp3_R10B_Baseline", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae", "pos", "color"]}},
    
    # 3. Tests redundancy. (Does hand-crafted texture like HOG add anything to deep CAE texture, or just noise?)
    {"name": "Exp3_R10C_Baseline_Plus_HOG", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae", "pos", "color", "hog"]}},
    
    # 4. Tests geometric noise. (Does knowing the actual shape/circularity of the superpixel help?)
    {"name": "Exp3_R10D_Baseline_Plus_Shape", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae", "pos", "color", "shape"]}},
    {"name": "Exp3_R11_all features", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "features": ["color", "pos", "hog", "cae", "shape", "lbp", "texture"]}},

    # --- EXPLORATION 4: Compression (Latent Vectors & Global Embeddings) ---
    {"name": "Exp4_R12_CAE_Dim32", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "cae_latent_dim": 32}},
    {"name": "Exp4_R13_CAE_Dim64", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "cae_latent_dim": 64}},
    {"name": "Exp4_R14_CAE_Dim128", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "cae_latent_dim": 128}},
    {"name": "Exp4_R15_Emb_Dim256", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "emb_dim": 256}},
    {"name": "Exp4_R16_Emb_Dim1024", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "emb_dim": 1024}},
    {"name": "Exp4_R16_1_Emb_Dim32", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "emb_dim": 32}},
    {"name": "Exp4_R16_2_Emb_Dim64", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "emb_dim": 64}},
    {"name": "Exp4_R16_3_Emb_Dim128", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "emb_dim": 128}},

    # --- EXPLORATION 5: Optimization Constraints (Pooling & Loss) ---
    {"name": "Exp5_R17_Pool_Mean", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "pooling_type": "mean"}},
    {"name": "Exp5_R18_Pool_Max", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "pooling_type": "max"}},
    {"name": "Exp5_R19_No_OrthoLoss", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "ortho_weight": 0.0}},

    # --- EXPLORATION 6: Granularity (Using Baseline Features) ---
    {"name": "Exp6_R20_Macro_50SP", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "seeds_num_superpixels": 50}},
    {"name": "Exp6_R21_Micro_500SP", "args": {"gnn_layers": 1, "gnn_type": "gatv2", "seeds_num_superpixels": 500, "batch_size": 128}},

    # --- EXPLORATION 7: Data Scaling (Adding Auxiliary Datasets) ---
    {"name": "Exp7_R22_Enriched_Scaling", "args": {
        "gnn_layers": 1, "gnn_type": "gatv2", 
        "split_datasets": ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "AmvrakikosTurtles", "ATRW", "LeopardID2022"]
    }},

    # --- EXPLORATION 8: Dedicated Single Species Models ---
    {"name": "Exp8_R23_Lynx_Only",       "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", "ortho_weight": 0.0,
        "split_datasets": ["LynxID2025"]}},
    {"name": "Exp8_R24_Salamander_Only", "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", "ortho_weight": 0.0,
        "split_datasets": ["SalamanderID2025"]}},
    {"name": "Exp8_R25_Turtle_Only",     "args": {"gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", "ortho_weight": 0.0,
        "split_datasets": ["SeaTurtleID2022"]}},

    # --- 1-LAYER PROBES (Testing if shallow networks overfit less on single species) ---
    {"name": "Exp8_R26_Lynx_Probe_1L", "args": {
        "gnn_layers": 1, "gnn_type": "gatv2", 
        "split_datasets": ["LynxID2025"], "ortho_weight": 0.0
    }},
    {"name": "Exp8_R27_Turtle_Probe_1L", "args": {
        "gnn_layers": 1, "gnn_type": "gatv2", 
        "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0
    }},
    {"name": "Exp8_R28_Salamander_Probe_1L", "args": {
        "gnn_layers": 1, "gnn_type": "gatv2", 
        "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0
    }},

    # # --- PURE TEXTURE (Testing if morphology alone drives identification) ---
    # {"name": "Exp8_R29_Lynx_PureTexture", "args": {
    #     "gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae"], 
    #     "split_datasets": ["LynxID2025"], "ortho_weight": 0.0
    # }},
    # {"name": "Exp8_R30_Turtle_PureTexture", "args": {
    #     "gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae"], 
    #     "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0
    # }},
    # {"name": "Exp8_R31_Salamander_PureTexture", "args": {
    #     "gnn_layers": 1, "gnn_type": "gatv2", "features": ["cae"], 
    #     "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0
    # }},

    # # --- HYBRID ROUTING (Testing if attention-based edges improve single-species clustering) ---
    # {"name": "Exp8_R32_Lynx_HybridEdges", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", 
    #     "edge_strategy": "hybrid", "k_neighbors": 5, 
    #     "split_datasets": ["LynxID2025"], "ortho_weight": 0.0
    # }},
    # {"name": "Exp8_R33_Turtle_HybridEdges", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", 
    #     "edge_strategy": "hybrid", "k_neighbors": 5, 
    #     "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0
    # }},
    # {"name": "Exp8_R34_Salamander_HybridEdges", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention", 
    #     "edge_strategy": "hybrid", "k_neighbors": 5, 
    #     "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0
    # }},

    # =====================================================================
    # --- EXPLORATION 9: TAXONOMIC HYPOTHESIS TESTING ---
    # =====================================================================

    # HYPOTHESIS 1: Amphibian Deformability
    # Tests if replacing rigid spatial edges with dynamic feature-based edges 
    # helps the GNN overcome the extreme bending/curling of Salamanders.
    # {"name": "Exp9_H1_Salamander_HybridEdges", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
    #     "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0,                      #Dublicate
    #     "edge_strategy": "hybrid", "k_neighbors": 5
    # }},

    # HYPOTHESIS 2: Amphibian Micro-Textures
    # Tests if increasing the superpixel resolution from 240 to 500 captures 
    # the tiny, high-frequency skin patterns that are currently getting blurred.
    # {"name": "Exp9_H2_Salamander_Micro500SP", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
    #     "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0,
    #     "seeds_num_superpixels": 500, "batch_size": 128  # Lower batch size to prevent OOM
    # }},

    # # HYPOTHESIS 3: The Ultimate Amphibian Fix
    # # Combines H1 and H2. If this yields the highest ARI, you have proven that 
    # # Amphibian Re-ID requires BOTH micro-textural resolution and flexible topology.
    # {"name": "Exp9_H3_Salamander_Hybrid_Micro", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
    #     "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0,
    #     "edge_strategy": "hybrid", "k_neighbors": 5,
    #     "seeds_num_superpixels": 500, "batch_size": 128
    # }},

    # # HYPOTHESIS 4: Reptilian Color Washing
    # # Tests if completely blinding the network to RGB color and forcing it to 
    # # rely purely on CAE texture patterns fixes the underwater caustics problem.
    # {"name": "Exp9_H4_Turtle_PureTexture", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
    #     "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0,
    #     "features": ["cae"]  # Removed "color" and "pos"
    # }},

    # # HYPOTHESIS 5: Reptilian Shape & Texture
    # # Tests if the network still needs spatial coordinates ("pos") alongside the 
    # # pure texture ("cae") to understand the layout of the turtle's scutes.
    # {"name": "Exp9_H5_Turtle_ShapeAndTexture", "args": {
    #     "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
    #     "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0,
    #     "features": ["cae", "pos"] # Blinds color, but keeps geometry
    # }},

    # =====================================================================
    # --- EXPLORATION 10: REGULARIZATION & DROPOUTS ---
    # =====================================================================
    
    # HYPOTHESIS: Heavy dropout prevents the network from memorizing the training set.
    {"name": "Exp10_R35_No_Dropouts", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "drop_node": 0.0, "drop_edge": 0.0, "drop_modality": 0.0
    }},
    # Isolate just the Modality Dropout to see if "blinding" features actually helps.
    {"name": "Exp10_R36_No_ModalityDrop", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "drop_modality": 0.0
    }},
    # Isolate just the Structural Dropout to see if forcing alternative message routing is needed.
    {"name": "Exp10_R37_No_GraphDrop", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "drop_node": 0.0, "drop_edge": 0.0
    }},

    # =====================================================================
    # --- EXPLORATION 11: (BACKGROUND ISOLATION) ---
    # =====================================================================
    
    # HYPOTHESIS: The performance gap between Lynx and amphibians/reptiles is not purely 
    # morphological, but caused by the GNN overfitting to background textures (water/mud) 
    # present in the cold-blooded datasets but absent in the pre-segmented Lynx dataset.
    
    {"name": "Exp11_R38_Turtle_Cutouts", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "split_datasets": ["SeaTurtleID2022"], "ortho_weight": 0.0,
        "use_cutouts": True  # Triggers the new LMDB key and directory shift
    }},
    {"name": "Exp11_R39_Salamander_Cutouts", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "split_datasets": ["SalamanderID2025"], "ortho_weight": 0.0,
        "use_cutouts": True
    }},
    
    {"name": "Exp11_R41_All_Species_Cutouts", "args": {
        "gnn_layers": 3, "gnn_type": "gatv2", "use_jk": True, "jk_mode": "attention",
        "ortho_weight": 1.0,  
        "use_cutouts": True
    }},
]

# ========================================================
# 5. STATE MACHINE UTILITIES
# ========================================================
def is_step_done(ckpt_dir, step_key):
    return os.path.exists(os.path.join(ckpt_dir, f".done_{step_key}"))

def mark_step_done(ckpt_dir, step_key):
    with open(os.path.join(ckpt_dir, f".done_{step_key}"), 'w') as f:
        f.write(f"Completed at {time.ctime()}")

def log_crash(ckpt_dir, step_name, error_msg):
    crash_path = os.path.join(ckpt_dir, "crash_report.txt")
    with open(crash_path, 'a') as f:
        f.write(f"[{time.ctime()}] FAILED AT: {step_name}\nERROR DETAILS:\n{error_msg}\n{'-'*50}\n")

def run_cmd(step_name, cmd_list, ckpt_dir, step_key):
    if is_step_done(ckpt_dir, step_key):
        print(f"{SUCCESS_LIME}[RESUME SKIP] {step_name} already completed.{RESET}")
        return

    print(f"{FOX_ORANGE}--- EXECUTING: {step_name} ---{RESET}")
    try:
        if cmd_list[0] == "python":
            cmd_list[0] = sys.executable
        cmd = [str(c) for c in cmd_list if c]
        subprocess.run(cmd, check=True)
        mark_step_done(ckpt_dir, step_key)
        print(f"{SUCCESS_LIME}[SUCCESS] {step_name} finished.{RESET}\n")
    except subprocess.CalledProcessError as e:
        print(f"{DANGER_RED}[ERROR] {step_name} CRASHED. Code: {e.returncode}.{RESET}")
        log_crash(ckpt_dir, step_name, f"Subprocess exited with code {e.returncode}.")
        raise e

def get_latest_checkpoint(directory):
    files = glob.glob(os.path.join(directory, "gnn", "*.pth"))
    return max(files, key=os.path.getmtime) if files else None

def dict_to_args(d):
    args = []
    for k, v in d.items():
        if v is True: args.append(f"--{k}")
        elif v is False or v is None: continue
        elif isinstance(v, list):
            args.append(f"--{k}")
            args.extend([str(x) for x in v])
        else:
            args.extend([f"--{k}", str(v)])
    return args

# ========================================================
# 6. EXECUTION LOGIC
# ========================================================
def main():
    os.makedirs("runs/ablations", exist_ok=True)
    os.makedirs(SHARED_CAE_DIR, exist_ok=True)
    os.environ["HF_TOKEN"] = "YOUR TOKEN HERE"

    print(f"{FOX_ORANGE}========================================================{RESET}")
    print(f"{FOX_ORANGE} ST-VGANN ABLATIONS {RESET}")
    print(f"{FOX_ORANGE}========================================================{RESET}\n")

    failed_runs = []

    for i, ablation in enumerate(ABLATIONS):
        run_name = ablation["name"]
        print(f"\n{DIRT_BROWN}========================================================{RESET}")
        print(f"{FOX_WHITE}[{i+1}/{len(ABLATIONS)}] STARTING RUN: {run_name}{RESET}")
        print(f"{DIRT_BROWN}========================================================{RESET}\n")

        config = BASE_CONFIG.copy()
        config.update(ablation["args"])
        ckpt_dir = f"runs/ablations/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(ckpt_dir, "run_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        merged_csv = os.path.join(ckpt_dir, "merged.csv")
        train_csv  = os.path.join(ckpt_dir, "train_split.csv")
        test_csv   = os.path.join(ckpt_dir, "test_split.csv")
        comp_test_csv = os.path.join(ckpt_dir, "competition_test.csv")
        if "Exp8" in run_name or "Exp9" in run_name or "Exp11" in run_name:
            baseline_test_csv = os.path.join("runs", "ablations", "Exp0_R00_Baseline", "test_split.csv")
            if os.path.exists(baseline_test_csv):
                test_csv = baseline_test_csv
        uses_cae = "cae" in config["features"]
        cae_name = f"cae_dim{config['cae_latent_dim']}_size{config['img_size']}_seeds{config['seeds_num_superpixels']}_v2"
        cae_weights_path = os.path.join(SHARED_CAE_DIR, f"{cae_name}.pth")

        try:
            # --- 0.0 MERGE ---
            if STEPS_TO_RUN.get("0.0_MERGE"):
                run_cmd(f"[{run_name}] STEP 0.0: Merge Datasets", 
                        ["python", "src/ml_utils/merge_datasets.py", "--reid_csv", REID_CSV, "--clef_csv", CLEF_CSV, "--out_csv", merged_csv], 
                        ckpt_dir, "0.0_MERGE")

            # --- 0 SPLIT ---
            if STEPS_TO_RUN.get("0_SPLIT"):
                split_cmd = ["python", "src/ml_utils/dataset_splitter.py", "--csv_path", merged_csv, "--save_dir", ckpt_dir]
                if config["split_datasets"]:
                    split_cmd.extend(["--datasets"] + config["split_datasets"])
                run_cmd(f"[{run_name}] STEP 0: Split Datasets", split_cmd, ckpt_dir, "0_SPLIT")

            # --- 2 TRAIN CAE ---
            if STEPS_TO_RUN.get("2_TRAIN_CAE") and uses_cae:
                if not os.path.exists(cae_weights_path):
                    run_cmd(f"[{run_name}] STEP 2: Train CAE", [
                        "python", "src/ml_utils/gnn/cae.py", "--root_dir", COMMON_ROOT,
                        "--csv_path", train_csv, "--epochs", 150, "--latent_dim", config["cae_latent_dim"],
                        "--save_dir", SHARED_CAE_DIR, "--checkpoint_dir", ckpt_dir,
                        "--model_name", cae_name, "--img_size", config["img_size"]
                    ], ckpt_dir, f"2_TRAIN_CAE_{cae_name}")
                else:
                    mark_step_done(ckpt_dir, f"2_TRAIN_CAE_{cae_name}")

            # --- 3 TRAIN GNN ---
            if STEPS_TO_RUN.get("3_TRAIN_GNN"):
                train_args = {k: v for k, v in config.items() if k not in ["split_datasets"]}
                train_cmd = ["python", "src/ml_utils/train_test_prototype.py", "--root_dir", COMMON_ROOT, 
                             "--csv_path", train_csv, "--checkpoint_dir", ckpt_dir] + dict_to_args(train_args)
                
                if uses_cae:
                    train_cmd.extend(["--cae_weights_path", cae_weights_path, "--cae_version", cae_name])

                if not is_step_done(ckpt_dir, "3_TRAIN_GNN"):
                    latest_ckpt = get_latest_checkpoint(ckpt_dir)
                    if latest_ckpt:
                        print(f"{SUCCESS_LIME}[INFO] Incomplete GNN training detected. Resuming from {os.path.basename(latest_ckpt)}...{RESET}")
                        train_cmd.extend(["--resume", latest_ckpt])

                run_cmd(f"[{run_name}] STEP 3: Train GNN", train_cmd, ckpt_dir, "3_TRAIN_GNN")

            # --- PREP EVALUATION ARGS ---
            eval_batch_gnn = 64
            eval_batch_wf = min(64, eval_batch_gnn)

            eval_args = {
                "root_dir": COMMON_ROOT, "img_size": config["img_size"], 
                "seeds_num_superpixels": config["seeds_num_superpixels"], 
                "seeds_num_levels": config["seeds_num_levels"], 
                "num_hog_bins": config["num_hog_bins"], 
                "checkpoints_dir": ckpt_dir, "features": config["features"],
                "leiden_k1": 12, "leiden_lambda": 0.2, "n_hops": config["n_hops"],
                "use_cutouts": config.get("use_cutouts", False)
            }
            if uses_cae:
                eval_args.update({"cae_weights_path": cae_weights_path, "cae_version": cae_name, "cae_latent_dim": config["cae_latent_dim"]})

            base_eval_cmd = ["python", "src/ml_utils/eval.py"] + dict_to_args(eval_args)

            # ==================================================
            # NORMAL EVALUATIONS (HARDCODED HDBSCAN)
            # ==================================================
            if STEPS_TO_RUN.get("4A_EVAL_KNOWN_GNN"):
                run_cmd(f"[{run_name}] STEP 4A: Known Domain GNN", base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, "--parallel_workers", 14], ckpt_dir, "4A_EVAL_KNOWN_GNN")

           # =================================================================
            # 4C: PER-SPECIES EVALUATION (FORCED CROSS-EVALUATION)
            # =================================================================
            if STEPS_TO_RUN.get("4C_EVAL_KNOWN_PER_SPECIES"):
                # Always grab the comprehensive test set from the Baseline
                baseline_test_csv = os.path.join("runs", "ablations", "Exp0_R00_Baseline", "test_split.csv")
                source_csv = baseline_test_csv if os.path.exists(baseline_test_csv) else test_csv
                df_test_full = pd.read_csv(source_csv, low_memory=False)
                
                species_datasets = {
                    "lynx": "LynxID2025",
                    "salamander": "SalamanderID2025",
                    "sea_turtle": "SeaTurtleID2022"
                }
                
                for sp_name, sp_ds in species_datasets.items():
                    df_sp = df_test_full[df_test_full['dataset'] == sp_ds]
                    if len(df_sp) == 0:
                        continue
                        
                    sp_csv_path = os.path.join(ckpt_dir, f"test_split_{sp_name}.csv")
                    df_sp.to_csv(sp_csv_path, index=False)
                    
                    # 1. Evaluate Raw Cosine (Matches .done_4C_EVAL_KNOWN_GNN_lynx_NO_RERANK)
                    run_cmd(f"[{run_name}] STEP 4C: Eval {sp_name.upper()} (Raw Cosine)", 
                            base_eval_cmd + ["--csv_path", sp_csv_path, "--batch_size", eval_batch_gnn, 
                                             "--disable_reranking", "--eval_suffix", f"_raw_cosine_species_{sp_name}"], 
                            ckpt_dir, f"4C_EVAL_KNOWN_GNN_{sp_name}_NO_RERANK")
                    
                    # 2. Evaluate Jaccard Reranked (Matches .done_4C_EVAL_KNOWN_GNN_lynx)
                    run_cmd(f"[{run_name}] STEP 4C: Eval {sp_name.upper()} (Jaccard)", 
                            base_eval_cmd + ["--csv_path", sp_csv_path, "--batch_size", eval_batch_gnn, 
                                             "--eval_suffix", f"_species_{sp_name}"], 
                            ckpt_dir, f"4C_EVAL_KNOWN_GNN_{sp_name}")

            if STEPS_TO_RUN.get("5A_EVAL_UNSEEN_GNN"):
                run_cmd(f"[{run_name}] STEP 5A: Unseen Domain GNN", base_eval_cmd + ["--csv_path", merged_csv, "--holdout_dataset", HOLDOUT_DATASET, "--batch_size", eval_batch_gnn, "--parallel_workers", 14], ckpt_dir, "5A_EVAL_UNSEEN_GNN")

            if STEPS_TO_RUN.get("6A_EVAL_MIXED_GNN"):
                run_cmd(f"[{run_name}] STEP 6A: Mixed Domain GNN", base_eval_cmd + ["--csv_path", merged_csv, "--base_test_csv", test_csv, "--holdout_dataset", HOLDOUT_DATASET, "--batch_size", eval_batch_gnn, "--parallel_workers", 14], ckpt_dir, "6A_EVAL_MIXED_GNN")

            if STEPS_TO_RUN.get("6A_EVAL_MIXED_GNN_NO_RERANK"):
                run_cmd(f"[{run_name}] STEP 6A (Raw): Mixed Domain No Reranking", 
                        base_eval_cmd + [
                            "--csv_path", merged_csv, 
                            "--base_test_csv", test_csv, 
                            "--holdout_dataset", HOLDOUT_DATASET, 
                            "--batch_size", eval_batch_gnn, 
                            "--parallel_workers", 14, 
                            "--disable_reranking", 
                            "--eval_suffix", "_raw_cosine"
                        ], 
                        ckpt_dir, "6A_EVAL_MIXED_GNN_NO_RERANK")

            if STEPS_TO_RUN.get("8_CROSS_EVAL_MATRIX"):
                pass # You can populate cross eval paths here if desired via explicit config keys.
            
            # ==================================================
            # OPTIMIZED EVALUATIONS (SEARCHING FOR BEST HDBSCAN)
            # ==================================================
            if STEPS_TO_RUN.get("4A_EVAL_KNOWN_GNN_OPT"):
                run_cmd(f"[{run_name}] STEP 4A (OPT): Known Domain GNN", base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, "--parallel_workers", 14, "--optimize_hdbscan", "--eval_suffix", "_opt"], ckpt_dir, "4A_EVAL_KNOWN_GNN_OPT")

            # =================================================================
            # 4C (OPT): PER-SPECIES EVALUATION WITH HDBSCAN OPTIMIZATION
            # =================================================================
            if STEPS_TO_RUN.get("4C_EVAL_KNOWN_PER_SPECIES_OPT"):
                species_datasets = ["lynx", "salamander", "sea_turtle"]
                
                for sp_name in species_datasets:
                    sp_csv_path = os.path.join(ckpt_dir, f"test_split_{sp_name}.csv")
                    if not os.path.exists(sp_csv_path):
                        continue
                        
                    # 3. Evaluate Jaccard + HDBSCAN Optimized (Matches .done_4C_EVAL_KNOWN_GNN_lynx_OPT)
                    run_cmd(f"[{run_name}] STEP 4C: Eval {sp_name.upper()} (Jaccard + Opt HDB)", 
                            base_eval_cmd + ["--csv_path", sp_csv_path, "--batch_size", eval_batch_gnn, 
                                             "--optimize_hdbscan", "--eval_suffix", f"_species_{sp_name}_opt"], 
                            ckpt_dir, f"4C_EVAL_KNOWN_GNN_{sp_name}_OPT")

            if STEPS_TO_RUN.get("5A_EVAL_UNSEEN_GNN_OPT"):
                run_cmd(f"[{run_name}] STEP 5A (OPT): Unseen Domain GNN", base_eval_cmd + ["--csv_path", merged_csv, "--holdout_dataset", HOLDOUT_DATASET, "--batch_size", eval_batch_gnn, "--parallel_workers", 14, "--optimize_hdbscan", "--eval_suffix", "_opt"], ckpt_dir, "5A_EVAL_UNSEEN_GNN_OPT")

            if STEPS_TO_RUN.get("6A_EVAL_MIXED_GNN_OPT"):
                run_cmd(f"[{run_name}] STEP 6A (OPT): Mixed Domain GNN", base_eval_cmd + ["--csv_path", merged_csv, "--base_test_csv", test_csv, "--holdout_dataset", HOLDOUT_DATASET, "--batch_size", eval_batch_gnn, "--parallel_workers", 14, "--optimize_hdbscan", "--eval_suffix", "_opt"], ckpt_dir, "6A_EVAL_MIXED_GNN_OPT")
            # ==================================================
            # POST-PROCESSING ABLATION (PURE GNN POWER)
            # ==================================================
            if STEPS_TO_RUN.get("4A_EVAL_KNOWN_NO_RERANK"):
                run_cmd(f"[{run_name}] STEP 4A (Raw): No Reranking", 
                        base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, 
                                         "--parallel_workers", 14, "--disable_reranking", 
                                         "--eval_suffix", "_raw_cosine"], 
                        ckpt_dir, "4A_EVAL_KNOWN_NO_RERANK")

            # if STEPS_TO_RUN.get("4C_EVAL_KNOWN_PER_SPECIES_NO_RERANK"):
            #     for species_dataset, species_tag in CLEF_SPECIES:
            #         if len(config.get("split_datasets", [])) == 1 and config["split_datasets"][0] != species_dataset:
            #             continue
            #         run_cmd(
            #             f"[{run_name}] STEP 4C (Raw): No Reranking [{species_tag}]",
            #             base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, 
            #                              "--parallel_workers", 14, "--holdout_dataset", species_dataset, 
            #                              "--disable_reranking", "--eval_suffix", f"_raw_cosine_species_{species_tag}"],
            #             ckpt_dir, f"4C_EVAL_KNOWN_GNN_{species_tag}_NO_RERANK"
            #         )

            if STEPS_TO_RUN.get("5A_EVAL_UNSEEN_GNN_NO_RERANK"):
                run_cmd(f"[{run_name}] STEP 5A (Raw): Unseen Domain No Reranking", 
                        base_eval_cmd + ["--csv_path", merged_csv, "--holdout_dataset", HOLDOUT_DATASET, 
                                         "--batch_size", eval_batch_gnn, "--parallel_workers", 14, 
                                         "--disable_reranking", "--eval_suffix", "_raw_cosine"], 
                        ckpt_dir, "5A_EVAL_UNSEEN_GNN_NO_RERANK")
                
            if STEPS_TO_RUN.get("4A_EVAL_KNOWN_OPT_NO_RERANK"):
                run_cmd(f"[{run_name}] STEP 4A (Raw+Opt): Naked Oracle Search", 
                        base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, 
                                         "--parallel_workers", 14, "--disable_reranking", 
                                         "--optimize_hdbscan", "--eval_suffix", "_raw_cosine_opt"], 
                        ckpt_dir, "4A_EVAL_KNOWN_OPT_NO_RERANK")
                
            if STEPS_TO_RUN.get("9_GENERATE_TSNE"):
                run_cmd(f"[{run_name}] STEP 9: Generate Top-K t-SNE", 
                        base_eval_cmd + ["--csv_path", test_csv, "--batch_size", eval_batch_gnn, 
                                         "--tsne_only", "--top_k_detailed", "3"], # Only plot top 3 to save time
                        ckpt_dir, "9_GENERATE_TSNE")
                
        except Exception as e:
            failed_runs.append(run_name)
            print(f"\n{DANGER_RED}[!] CRITICAL EXCEPTION CAUGHT IN RUN {run_name}.{RESET}")
            print(f"{DANGER_RED}Logging to crash_report.txt and skipping to the next experiment...{RESET}\n")
            log_crash(ckpt_dir, "Orchestrator Pipeline Exception", traceback.format_exc())
            continue

    print("\n========================================================")
    print(" ABLATION SUITE COMPLETE")
    print("========================================================")
    if failed_runs:
        print(f"{DANGER_RED}The following runs crashed and were skipped. Check their respective 'crash_report.txt' files:{RESET}")
        for fr in failed_runs:
            print(f" - {fr}")
    else:
        print(f"{SUCCESS_LIME}All scheduled ablations executed and evaluated successfully!{RESET}")

if __name__ == "__main__":
    main()