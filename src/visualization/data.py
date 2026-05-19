import os
import glob
import warnings
import pandas as pd
import numpy as np

from config import PRIMARY_METRIC, SECONDARY_METRIC, GUARANTEED_COLS

# ============================================================================
# LOADING
# ============================================================================

def load_all_results(base_dir: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "**", "benchmark_stats.csv")
    files   = glob.glob(pattern, recursive=True)
    print(f"   Found {len(files)} benchmark_stats.csv files.")

    frames = []
    for f in files:
        parts = f.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue

        run_name = parts[-3]
        eval_dir = parts[-2]

        if "analysis_reports" in run_name:
            continue

        try:
            df = pd.read_csv(f, low_memory=False)
            if df.empty:
                continue
            df["Run_Name"] = run_name
            df["Eval_Dir"] = eval_dir
            df["Source_File"] = f  # <--- NEW: Forces the script to track the absolute path
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"   [!] Could not read {f}: {exc}")

    if not frames:
        return pd.DataFrame()

    master = pd.concat(frames, ignore_index=True)

    for col in GUARANTEED_COLS:
        if col not in master.columns:
            master[col] = np.nan

    numeric_cols = GUARANTEED_COLS + [PRIMARY_METRIC, SECONDARY_METRIC]
    for col in numeric_cols:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    print(f"   Loaded {len(master):,} checkpoint rows from "
          f"{master['Run_Name'].nunique()} distinct runs.")
    return master


# ============================================================================
# PARSING EVAL-DIR NAMES
# ============================================================================
def parse_eval_dir(d: str, run_name: str):
    d = d.lower()
    
    # 1. Safely extract the domain using flexible keywords
    if "lynx" in d and "species" in d:
        domain = "Isolated: Lynx"
    elif "salamander" in d and "species" in d:
        domain = "Isolated: Salamander"
    elif "turtle" in d and "species" in d:
        domain = "Isolated: Turtle"
    elif "mixed_holdout" in d:
        domain = "Mixed (Known + Unseen)"
    elif "dataset_holdout" in d:
        domain = "Unseen (OOD)"
    else:
        domain = "Known (Mixed Species)"

    # 2. Extract reranking and optimisation status
    reranking = "disabled" if "raw_cosine" in d else "enabled"
    tokens = d.replace("-", "_").split("_")
    optimised = "opt" in tokens

    return domain, reranking, optimised

# ============================================================================
# PEAK PERFORMANCE
# ============================================================================

def get_peak_performance(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    group_cols = ["Run_Name", "Domain", "Reranking", "Optimised"]
    
    df_clean = df[df["Domain"] != "Competition (Unlabeled)"].copy()
    df_clean = df_clean.dropna(subset=[metric])

    if df_clean.empty:
        return pd.DataFrame()

    idx  = df_clean.groupby(group_cols)[metric].idxmax()
    peak = df_clean.loc[idx].reset_index(drop=True)
    
    return peak