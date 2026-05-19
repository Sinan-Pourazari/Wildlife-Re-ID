import os
import pandas as pd # Ensure pandas is imported
from config import RESULTS_BASE, OUTPUT_DIR, PRIMARY_METRIC
from data import load_all_results, parse_eval_dir, get_peak_performance

from tables import generate_all_family_tables, generate_all_global_tables
from plots import set_style, generate_all_family_plots, generate_all_global_plots

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_dir = os.path.join(OUTPUT_DIR, "Global_Summary")
    os.makedirs(global_dir, exist_ok=True)

    set_style()

    print("==================================================")
    print(" Ablation Analysis")
    print("==================================================")

    # ── 1. Load & parse ───────────────────────────────────────────────────────
    master = load_all_results(RESULTS_BASE)
    if master.empty:
        print("\n[!] No CSV data found. Check RESULTS_BASE in config.py.")
        return

    # Pass BOTH the directory and the run name to the updated parser
    parsed             = master.apply(lambda row: parse_eval_dir(row["Eval_Dir"], row["Run_Name"]), axis=1)
    master["Domain"]   = parsed.apply(lambda x: x[0])
    master["Reranking"]= parsed.apply(lambda x: x[1])
    master["Optimised"]= parsed.apply(lambda x: x[2])
    master             = master.dropna(subset=[PRIMARY_METRIC])

    # Get the peak performance (best epoch per configuration)
    peak = get_peak_performance(master, PRIMARY_METRIC)
    
    master_csv_path = os.path.join(OUTPUT_DIR, "master_peak_performance.csv")
    peak.to_csv(master_csv_path, index=False)
    
    print(f"Peak table: {len(peak)} discrete evaluations across "
          f"{peak['Run_Name'].nunique()} runs.")
    print(f"Consolidated data saved to: {master_csv_path}\n")

    # ── 2. Per-family reports ─────────────────────────────────────────────────
    print("Generating per-family tables & plots …")
    generate_all_family_tables(peak, OUTPUT_DIR)
    generate_all_family_plots(master, peak, OUTPUT_DIR)

    # ── 3. Global summaries ───────────────────────────────────────────────────
    print("Generating global summaries …")
    generate_all_global_tables(peak, global_dir)
    generate_all_global_plots(master, peak, global_dir)


    print(f"\n[+] Done.  All outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()