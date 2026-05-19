import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

def get_paper_section(run_name: str) -> str:
    if run_name.startswith("Exp4_"):
        if "CAE" in run_name: return "5.6. Ablation 4: CAE Latent Dimension"
        if "Emb" in run_name: return "5.7. Ablation 5: Output Embedding Dimension"
    
    mapping = {
        "Exp0": "5.2. Baseline Configuration",
        "Exp1": "5.3. Ablation 1: Edge Construction Method",
        "Exp2": "5.4. Ablation 2: Topological Depth",
        "Exp3": "5.5. Ablation 3: Feature Composition",
        "Exp5": "5.8. Ablation 6: Pooling Method and Orthogonality",
        "Exp6": "5.9. Ablation 7: Superpixel Granularity",
        "Exp7": "5.10. Ablation 8: Extended Training Data",
        "Exp8": "5.11. Ablation 9: Specialist Models",
        "Exp9": "Taxonomic Hypothesis Testing",
        "Exp10": "5.12. Ablation 10: Regularization Approaches",
        "Exp11": "5.13. Exp 11: Background Masking"
    }
    parts = run_name.split("_")
    return mapping.get(parts[0], "Other Experiments")

def get_human_readable_variant(run_name: str) -> str:
    DEEP_BASELINE_RUN    = "Exp0_R00_Baseline"
    SHALLOW_BASELINE_RUN = "Exp2_R07_GATv2_1L"

    if run_name == DEEP_BASELINE_RUN:
        return "Baseline (3L Spatial)"
    if run_name == SHALLOW_BASELINE_RUN:
        return "Shallow Baseline (1L Spatial)"
        
    parts = run_name.split("_")
    
    if len(parts) >= 3 and (parts[1].startswith("R") or parts[1].startswith("H")):
        if parts[2].isdigit() and len(parts) > 3:
            clean_name = " ".join(parts[3:])
        else:
            clean_name = " ".join(parts[2:])
        return clean_name.replace("_", " ")
        
    clean_name = " ".join(parts[1:]) if len(parts) > 1 else run_name
    return clean_name.replace("_", " ")

def get_layer_depth(human_name: str) -> str:
    if "1L" in human_name:
        return "1-Layer"
    elif "2L" in human_name:
        return "2-Layer"
    else:
        return "3-Layer (Default)"

def extract_sort_keys(run_name: str):
    exp_match = re.search(r'Exp(\d+)', run_name)
    exp_num = int(exp_match.group(1)) if exp_match else 999
    r_match = re.search(r'_([RH])(\d+)', run_name)
    r_type = r_match.group(1) if r_match else 'Z'
    r_num = int(r_match.group(2)) if r_match else 999
    return (exp_num, r_type, r_num, run_name)

def format_time(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

# ============================================================================
# CRAWLER & AGGREGATOR
# ============================================================================
def calculate_ablation_training_times(base_folder, output_base_dir, file_pattern="*.pth", outlier_multiplier=4.0):
    base_path = Path(base_folder)
    experiment_times = []

    checkpoints = list(base_path.rglob(file_pattern))
    
    grouped_files = {}
    for cp in checkpoints:
        try:
            exp_name = str(cp.parent.relative_to(base_path))
        except ValueError:
            exp_name = cp.parent.name
            
        if exp_name not in grouped_files:
            grouped_files[exp_name] = []
        grouped_files[exp_name].append(cp)

    print(f"Found {len(grouped_files)} experiment groups containing checkpoints...\n")

    time_pattern = re.compile(r'_(\d{8}_\d{6})\.pth$')

    for exp_name, files in grouped_files.items():
        if len(files) < 2:
            continue
            
        run_folder = Path(exp_name).parts[0]
        paper_section = get_paper_section(run_folder)
        
        if paper_section == "Other Experiments":
            continue
            
        times = []
        for f in files:
            match = time_pattern.search(f.name)
            if match:
                time_str = match.group(1)
                dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                times.append(dt.timestamp())
            else:
                times.append(os.path.getmtime(f))
        
        times.sort()
        deltas = np.diff(times)
        
        if len(deltas) == 0:
            continue

        median_delta = np.median(deltas)
        outlier_mask = deltas > (median_delta * outlier_multiplier)
        num_interruptions = np.sum(outlier_mask)
        clean_deltas = np.where(outlier_mask, median_delta, deltas)
        total_seconds = np.sum(clean_deltas)
        
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        human_name = get_human_readable_variant(run_folder)
        median_epoch = median_delta / 2.0
        
        experiment_times.append({
            "Run ID": run_folder,
            "Paper Section": paper_section,
            "Experiment Variant": human_name,
            "Layer Depth": get_layer_depth(human_name),
            "Checkpoints": len(files),
            "Interruptions": num_interruptions,
            "Median Epoch (s)": round(median_epoch, 2),
            "Est. Compute Time": f"{hours}h {minutes}m",
            "_raw_epoch_s": median_epoch,
            "_total_seconds": total_seconds
        })

    if not experiment_times:
        print("No valid official experiments found with 2+ checkpoints.")
        return
        
    df = pd.DataFrame(experiment_times)
    
    df['sort_key'] = df['Run ID'].apply(extract_sort_keys)
    df = df.sort_values(by='sort_key').drop(columns=['sort_key', 'Run ID'])
    
    # Calculate Total Cumulative Time
    grand_total_s = df["_total_seconds"].sum()
    g_hours = int(grand_total_s // 3600)
    g_minutes = int((grand_total_s % 3600) // 60)
    g_days = g_hours // 24
    g_hours_rem = g_hours % 24
    
    if g_days > 0:
        total_time_str = f"{g_days}d {g_hours_rem}h {g_minutes}m"
    else:
        total_time_str = f"{g_hours}h {g_minutes}m"
        
    df_total = pd.DataFrame([{"Metric": "Total Cumulative Training Time", "Value": total_time_str}])
    
    df_main = df[["Paper Section", "Experiment Variant", "Median Epoch (s)", "Est. Compute Time"]].copy()
    
    # Compile the consolidated averages table
    avg_data = []
    
    overall_avg_epoch = df["_raw_epoch_s"].mean()
    overall_avg_total = df["_total_seconds"].mean()
    
    avg_data.append({
        "Category Type": "Global",
        "Category": "All Experiments",
        "Avg. Epoch Time (s)": round(overall_avg_epoch, 2),
        "Avg. Total Time": format_time(overall_avg_total)
    })
    
    for depth, group in df.groupby("Layer Depth"):
        avg_data.append({
            "Category Type": "Layer Depth",
            "Category": depth,
            "Avg. Epoch Time (s)": round(group["_raw_epoch_s"].mean(), 2),
            "Avg. Total Time": format_time(group["_total_seconds"].mean())
        })
        
    for section, group in df.groupby("Paper Section", sort=False):
        avg_data.append({
            "Category Type": "Paper Section",
            "Category": section,
            "Avg. Epoch Time (s)": round(group["_raw_epoch_s"].mean(), 2),
            "Avg. Total Time": format_time(group["_total_seconds"].mean())
        })
        
    df_averages = pd.DataFrame(avg_data)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_base_dir) / f"report_{timestamp_str}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    def export_table(dataframe, filename, use_longtable=False, col_format=None):
        md_path = report_dir / f"{filename}.md"
        tex_path = report_dir / f"{filename}.tex"
        
        dataframe.to_markdown(md_path, index=False)
        
        kwargs_style = {"hrules": True}
        
        if use_longtable:
            kwargs_style["environment"] = "longtable"
            
        if col_format:
            kwargs_style["column_format"] = col_format
            
        try:
            styler = dataframe.style.hide(axis="index").format(precision=2)
            
            tex_str = styler.to_latex(**kwargs_style)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_str)
        except (AttributeError, TypeError):
            dataframe.to_latex(tex_path, index=False)
            
    export_table(df_main, "1_main_training_times", use_longtable=True, 
                 col_format="p{0.25\\textwidth} p{0.35\\textwidth} c c")
                 
    export_table(df_averages, "2_aggregated_averages", 
                 use_longtable=True,col_format="p{0.2\\textwidth} p{0.4\\textwidth} c c")
                 
    export_table(df_total, "3_total_compute_time",use_longtable=False, col_format="p{0.5\\textwidth} c")

    print(f"Generated Markdown and LaTeX tables in: {report_dir}\n")
    print(f"TOTAL CUMULATIVE COMPUTE TIME: {total_time_str}\n")

if __name__ == "__main__":
    ABLATION_DIRECTORY = r"./runs/ablations/" 
    OUTPUT_DIRECTORY = r".\runs\ablations\analysis_reports"
    
    calculate_ablation_training_times(
        base_folder=ABLATION_DIRECTORY, 
        output_base_dir=OUTPUT_DIRECTORY,
        file_pattern="*.pth", 
        outlier_multiplier=4.0
    )