import pandas as pd

# ============================================================================
# PATHS
# ============================================================================
RESULTS_BASE = "runs/ablations"
OUTPUT_DIR   = "runs/ablations/analysis_reports"

# ============================================================================
# METRICS
# ============================================================================
PRIMARY_METRIC   = "Baseline ARI"   # AnimalCLEF competition clustering metric
SECONDARY_METRIC = "mAP (%)"        # Re-ID retrieval backup

ALL_RETRIEVAL_METRICS  = ["Rank-1 (%)", "Rank-5 (%)", "Rank-10 (%)", "mAP (%)"]
ALL_CLUSTER_METRICS    = ["Baseline ARI", "Baseline NMI", "Baseline IDs"]
ALL_OPENSET_METRICS    = ["BaKS", "BAUS", "H-Score"]
ALL_SPECIES_METRICS    = ["Species Acc (%)", "Species BAcc (%)"]
ALL_HDBSCAN_METRICS    = ["Opt Eps", "Opt MinCls"]

# Columns guaranteed to be read from benchmark_stats.csv
GUARANTEED_COLS = (
    ["Epoch", "Model Name"]
    + ALL_RETRIEVAL_METRICS
    + ALL_CLUSTER_METRICS
    + ALL_OPENSET_METRICS
    + ALL_SPECIES_METRICS
    + ALL_HDBSCAN_METRICS
)

# ============================================================================
# BASELINES
# ============================================================================
DEEP_BASELINE_RUN    = "Exp0_R00_Baseline"
SHALLOW_BASELINE_RUN = "Exp2_R07_GATv2_1L"

# ============================================================================
# EXPERIMENT FAMILIES
# ============================================================================
ORDERED_FAMILIES = [
    "Exp0", "Exp1", "Exp2", "Exp3",
    "Exp4_CAE", "Exp4_Emb", "Exp5", "Exp6",
    "Exp7", "Exp8", "Exp9", "Exp10", "Exp11"
]

FAMILY_NAMES = {
    "Exp0":     "Baseline (Control Group)",
    "Exp1":     "Graph Construction (Edge Strategies)",
    "Exp2":     "GNN Depth & Message Routing",
    "Exp3":     "Node Feature Vocabulary",
    "Exp4_CAE": "Compression (CAE Latent Dimension)",
    "Exp4_Emb": "Compression (Global Embedding Dimension)",
    "Exp5":     "Pooling & Orthogonality Constraint",
    "Exp6":     "Superpixel Granularity",
    "Exp7":     "Training Data Scaling",
    "Exp8":     "Single-Species Specialists",
    "Exp9":     "Taxonomic Hypothesis Testing",
    "Exp10":    "Regularisation & Dropout",
    "Exp11":    "Background Removal (Cutouts)",
}

# ============================================================================
# UTILITY FUNCTIONS (DYNAMIC PARSERS)
# ============================================================================

def get_family_prefix(run_name: str) -> str:
    """Returns the base family prefix for a given run (e.g. Exp4_CAE)."""
    if run_name.startswith("Exp4_") and "CAE" in run_name: return "Exp4_CAE"
    if run_name.startswith("Exp4_") and "Emb" in run_name: return "Exp4_Emb"
    
    parts = run_name.split("_")
    return parts[0] if len(parts) > 0 else "Unknown"


def get_target_runs(df: pd.DataFrame, family_key: str) -> list[str]:
    """Extracts all runs belonging to a specific experiment family."""
    df = df.dropna(subset=["Run_Name"]).copy()
    
    if family_key == "Exp4_CAE":
        mask = df["Run_Name"].str.startswith("Exp4_") & df["Run_Name"].str.contains("CAE")
    elif family_key == "Exp4_Emb":
        mask = df["Run_Name"].str.startswith("Exp4_") & df["Run_Name"].str.contains("Emb")
    elif family_key == "Exp11":
        # INJECT THE LYNX RUN
        mask = df["Run_Name"].str.startswith("Exp11_") | df["Run_Name"].str.contains("Exp8_R23")
        runs = df.loc[mask, "Run_Name"].unique().tolist()
        
        # CUSTOM SORT ORDER FOR EXP 11 PLOT
        def get_sort_weight(run_name):
            if "Baseline" in run_name or "R00" in run_name: return 0
            if "R38" in run_name: return 1  # Turtle
            if "R39" in run_name: return 2  # Salamander
            if "R23" in run_name: return 3  # Lynx (Now next to Salamander!)
            if "R41" in run_name: return 4  # All Cutouts
            return 5
            
        return sorted(runs, key=get_sort_weight)
    else:
        mask = df["Run_Name"].str.startswith(family_key + "_")
        
    return sorted(df.loc[mask, "Run_Name"].unique().tolist())

def get_human_readable_variant(run_name: str) -> str:
    """Dynamically parses complex run names into clean, readable labels."""
    if run_name == DEEP_BASELINE_RUN:
        return "Baseline (3L Spatial)"
    if run_name == SHALLOW_BASELINE_RUN:
        return "Shallow Baseline (1L Spatial)"
        
    parts = run_name.split("_")
    
    # Standard format: ExpX_RXX_Variant_Name or ExpX_RXXb_Variant_Name
    if len(parts) >= 3 and (parts[1].startswith("R") or parts[1].startswith("H")):
        # If the 3rd part is just a sub-run number (e.g. the '1' in Exp4_R16_1_Emb_Dim32), skip it
        if parts[2].isdigit() and len(parts) > 3:
            clean_name = " ".join(parts[3:])
        else:
            clean_name = " ".join(parts[2:])
        return clean_name.replace("_", " ")
        
    # Fallback for weirdly named runs
    clean_name = " ".join(parts[1:]) if len(parts) > 1 else run_name
    return clean_name.replace("_", " ")


def is_shallow_probe(run_name: str) -> bool:
    """Detects if an architecture is intentionally shallow for baseline comparisons."""
    return "1L" in run_name or "Probe" in run_name


def choose_baseline(runs_in_family: list[str]) -> tuple[str, str]:
    """Selects the most appropriate reference baseline for a given family of experiments."""
    if DEEP_BASELINE_RUN in runs_in_family:
        return DEEP_BASELINE_RUN, get_human_readable_variant(DEEP_BASELINE_RUN)
    if SHALLOW_BASELINE_RUN in runs_in_family:
        return SHALLOW_BASELINE_RUN, get_human_readable_variant(SHALLOW_BASELINE_RUN)
    
    # Fallback to the overarching baseline if no internal baseline exists
    return DEEP_BASELINE_RUN, get_human_readable_variant(DEEP_BASELINE_RUN)