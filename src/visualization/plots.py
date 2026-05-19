# plots.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.colors as mcolors
from config import (
    PRIMARY_METRIC, SECONDARY_METRIC,
    DEEP_BASELINE_RUN, SHALLOW_BASELINE_RUN,
    ORDERED_FAMILIES, FAMILY_NAMES,
    get_target_runs, get_human_readable_variant,
    choose_baseline, is_shallow_probe, get_family_prefix,
)

SPECIES_DOMAINS = ["Isolated: Lynx", "Isolated: Salamander", "Isolated: Turtle"]
SPECIES_COLOURS = {
    "Isolated: Lynx":       "#4C72B0",
    "Isolated: Salamander": "#4C72B0",
    "Isolated: Turtle":     "#4C72B0",
}

# ============================================================================
# STYLE
# ============================================================================

def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    plt.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "DejaVu Sans"],
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.titlepad":         10,
        "figure.dpi":            150,
        'grid.alpha': 0.3,       # Controls transparency (0 is invisible, 1 is opaque)
        'grid.linewidth': 0.5,   # Making lines thinner also makes them less intrusive
        'grid.color': '#e0e0e0'  
    })

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _hbar(ax: plt.Axes, y: list, x: list, colours: list | str = "steelblue", xlim_pad: float = 0.08) -> None:
    """Horizontal bar chart with inline value labels, no overflow."""
    bars = ax.barh(y, x, color=colours, edgecolor="white", linewidth=0.4)
    x_max = max((v for v in x if pd.notna(v)), default=0)
    ax.set_xlim(0, x_max + xlim_pad)
    for bar, val in zip(bars, x):
        if pd.notna(val) and val > 0:
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", fontsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

def _save_fig(fig: plt.Figure, path: str) -> None:
    # bbox_inches="tight" ensures the newly moved legends aren't cropped out
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _filter_peak(peak: pd.DataFrame, domain: str = "Known (Mixed Species)", rerank: str = "enabled", opt: bool = False) -> pd.DataFrame:
    return peak[(peak["Domain"] == domain) & (peak["Reranking"] == rerank) & (peak["Optimised"] == opt)]

# ============================================================================
# CONSTANTS & COLOURS
# ============================================================================
VAR_POS     = "#55A868"   # green  – better than baseline
VAR_NEG     = "#C44E52"   # red    – worse  than baseline
VAR_BASE    = "#FFAE00"   # gold   – DEEP BASELINE absolute value
VAR_SHALLOW = "#F4D03F"   # yellow - SHALLOW BASELINE absolute value

# ============================================================================
# FAMILY BAR CHART (Known vs Unseen OOD vs Mixed)
# ============================================================================

def plot_family_generalisation(peak: pd.DataFrame, family_key: str, fam_dir: str) -> None:
    runs = get_target_runs(peak, family_key)
    if not runs: return

    baseline_run, baseline_label = choose_baseline(list(runs))
    rows = []
    
    for run in (list(runs) + ([baseline_run] if baseline_run not in runs else [])):
        for domain, dm_label in [("Known (Mixed Species)", "Known"),
                                 ("Unseen (OOD)", "OOD (Holdout)"),
                                 ("Mixed (Known + Unseen)", "Mixed")]:
            sub = _filter_peak(peak, domain)
            sub = sub[sub["Run_Name"] == run]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            if pd.notna(ari):
                rows.append({
                    "run":    run,
                    "label":  baseline_label if run == baseline_run else get_human_readable_variant(run),
                    "domain": dm_label,
                    "ari":    ari,
                })

    df = pd.DataFrame(rows)
    if df.empty: return

    order_df = df[df["domain"] == "Known"].sort_values("ari", ascending=False)
    order = order_df["label"].tolist() if not order_df.empty else df["label"].unique().tolist()
    
    seen = set()
    order = [x for x in order if not (x in seen or seen.add(x))]

    fig_h  = max(4.5, len(order) * 0.7)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(data=df, y="label", x="ari", hue="domain", order=order,
                palette={"Known": "#4C72B0", "OOD (Holdout)": "#C44E52", "Mixed": "#DD8452"}, dodge=True, ax=ax)

    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if pd.notna(w) and w > 0.005:
                ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, df["ari"].max() + 0.12)
    ax.set_xlabel("Adjusted Rand Index (ARI)")
    ax.set_ylabel("")
    ax.set_title(f"{FAMILY_NAMES.get(family_key, family_key)}: Distributional Shift Robustness", fontweight="bold")
    
    # Legend centered at the bottom
    ax.legend(title="Evaluation Distribution", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_generalisation_bar.png"))

# ============================================================================
# FAMILY RERANKING IMPACT
# ============================================================================

def plot_family_reranking(peak: pd.DataFrame, family_key: str, fam_dir: str) -> None:
    runs = get_target_runs(peak, family_key)
    if not runs: return

    baseline_run, baseline_label = choose_baseline(list(runs))
    rows = []
    
    for run in (list(runs) + ([baseline_run] if baseline_run not in runs else [])):
        for rerank_val, lbl in [("disabled", "Raw Cosine Distance"), ("enabled", "k-Reciprocal Jaccard")]:
            sub = _filter_peak(peak, "Known (Mixed Species)", rerank=rerank_val, opt=False)
            sub = sub[sub["Run_Name"] == run]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            if pd.notna(ari):
                rows.append({"run": run, "label": baseline_label if run == baseline_run else get_human_readable_variant(run), "Condition": lbl, "ari": ari})

    df = pd.DataFrame(rows)
    if df.empty or df["Condition"].nunique() < 2: return

    order_df = df[df["Condition"] == "k-Reciprocal Jaccard"].sort_values("ari", ascending=False)
    order = order_df["label"].tolist() if not order_df.empty else df["label"].unique().tolist()
    
    seen = set()
    order = [x for x in order if not (x in seen or seen.add(x))]

    fig_h = max(4.5, len(order) * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(data=df, y="label", x="ari", hue="Condition", order=order,
                palette={"Raw Cosine Distance": "#C44E52", "k-Reciprocal Jaccard": "#4C72B0"}, dodge=True, ax=ax)

    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if pd.notna(w) and w > 0.005:
                ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, df["ari"].max() + 0.12)
    ax.set_xlabel("Adjusted Rand Index (ARI)")
    ax.set_ylabel("")
    ax.set_title(f"{FAMILY_NAMES.get(family_key, family_key)}: Post-Processing Performance Delta", fontweight="bold")
    
    # Legend centered at the bottom
    ax.legend(title="Distance Metric", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_reranking_impact.png"))

# ============================================================================
# FAMILY HDBSCAN OPTIMIZATION
# ============================================================================

def plot_family_hdbscan(peak: pd.DataFrame, family_key: str, fam_dir: str) -> None:
    runs = get_target_runs(peak, family_key)
    if not runs: return

    baseline_run, baseline_label = choose_baseline(list(runs))
    rows = []
    
    for run in (list(runs) + ([baseline_run] if baseline_run not in runs else [])):
        for opt_val, lbl in [(False, "Static HDBSCAN"), (True, "Optimized HDBSCAN")]:
            sub = _filter_peak(peak, "Known (Mixed Species)", rerank="enabled", opt=opt_val)
            sub = sub[sub["Run_Name"] == run]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            if pd.notna(ari):
                rows.append({"run": run, "label": baseline_label if run == baseline_run else get_human_readable_variant(run), "Condition": lbl, "ari": ari})

    df = pd.DataFrame(rows)
    if df.empty or df["Condition"].nunique() < 2: return

    order_df = df[df["Condition"] == "Optimized HDBSCAN"].sort_values("ari", ascending=False)
    order = order_df["label"].tolist() if not order_df.empty else df["label"].unique().tolist()
    
    seen = set()
    order = [x for x in order if not (x in seen or seen.add(x))]

    fig_h = max(4.5, len(order) * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(data=df, y="label", x="ari", hue="Condition", order=order,
                palette={"Static HDBSCAN": "#4C72B0", "Optimized HDBSCAN": "#D4A017"}, dodge=True, ax=ax)

    for container in ax.containers:
        for bar in container:
            w = bar.get_width()
            if pd.notna(w) and w > 0.005:
                ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, df["ari"].max() + 0.12)
    ax.set_xlabel("Adjusted Rand Index (ARI)")
    ax.set_ylabel("")
    ax.set_title(f"{FAMILY_NAMES.get(family_key, family_key)}: Hyperparameter Optimization Delta", fontweight="bold")
    
    # Legend centered at the bottom
    ax.legend(title="Clustering Regimen", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_hdbscan_impact.png"))

# ============================================================================
# FAMILY LEARNING CURVES
# ============================================================================

def plot_family_learning_curves(master: pd.DataFrame, peak: pd.DataFrame, family_key: str, fam_dir: str) -> None:
    runs = get_target_runs(master, family_key)
    if not runs: return

    baseline_run, baseline_label = choose_baseline(list(runs))

    base_eval_row = _filter_peak(peak, "Known (Mixed Species)")
    base_eval_row = base_eval_row[base_eval_row["Run_Name"] == baseline_run]
    base_eval_dir = base_eval_row["Eval_Dir"].iloc[0] if not base_eval_row.empty else None

    all_runs = list(runs) + ([baseline_run] if baseline_run not in runs else [])

    data = master[
        master["Run_Name"].isin(all_runs) &
        (master["Domain"] == "Known (Mixed Species)") &
        (master["Reranking"] == "enabled") &
        (~master["Optimised"])
    ].dropna(subset=[PRIMARY_METRIC, "Epoch"]).copy()

    if base_eval_dir is not None:
        base_mask  = data["Run_Name"] == baseline_run
        dir_mask   = data["Eval_Dir"] == base_eval_dir
        data = data[~base_mask | dir_mask]

    if data.empty: return

    data["Variant"] = data["Run_Name"].apply(lambda r: baseline_label if r == baseline_run else get_human_readable_variant(r))

    n_lines = data["Variant"].nunique()
    if n_lines > 8:
        top_runs = data.groupby("Run_Name")[PRIMARY_METRIC].max().nlargest(7).index.tolist()
        if baseline_run not in top_runs: top_runs.append(baseline_run)
        data = data[data["Run_Name"].isin(top_runs)]
        n_lines = data["Variant"].nunique()

    fig, ax = plt.subplots(figsize=(11, 5))
    palette  = sns.color_palette("tab10", n_lines)
    variants = data["Variant"].unique().tolist()
    col_map  = {v: palette[i] for i, v in enumerate(variants)}

    for variant, grp in data.groupby("Variant"):
        grp = grp.groupby("Epoch")[PRIMARY_METRIC].max().reset_index()
        grp = grp.sort_values("Epoch")
        lw  = 2.5 if variant == baseline_label else 1.6
        ls  = "--" if variant == baseline_label else "-"
        ax.plot(grp["Epoch"], grp[PRIMARY_METRIC], label=variant, color=col_map[variant], linewidth=lw, linestyle=ls, alpha=0.9)

    ax.set_xlabel("Training Iteration (Epoch)")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title(f"{FAMILY_NAMES.get(family_key, family_key).replace('§ ', '')}: Validation ARI Convergence", fontweight="bold")
    
    # Legend centered at the bottom (dynamically adjusting columns based on line count)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(4, n_lines), fontsize=9, framealpha=0.8)
    
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_learning_curve.png"))

# ============================================================================
# FAMILY SPECIES BREAKDOWN 
# ============================================================================

def plot_family_species(peak: pd.DataFrame, family_key: str, fam_dir: str) -> None:
    runs = get_target_runs(peak, family_key)
    if not runs: return

    baseline_run, baseline_label = choose_baseline(list(runs))
    all_runs = list(runs) + ([baseline_run] if baseline_run not in runs else [])

    panels: dict[str, pd.DataFrame] = {}
    for domain in SPECIES_DOMAINS:
        rows = []
        for run in all_runs:
            sub = peak[(peak["Run_Name"] == run) & (peak["Domain"] == domain) & (peak["Reranking"] == "enabled") & (~peak["Optimised"])]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            if pd.notna(ari):
                rows.append({"label": baseline_label if run == baseline_run else get_human_readable_variant(run), "ari": ari, "run": run})
        if rows: panels[domain] = pd.DataFrame(rows).sort_values("ari", ascending=True)

    if not panels: return

    n_panels  = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, max(4, len(all_runs) * 0.4)), squeeze=False)
    axes = axes[0]

    for ax, (domain, df_p) in zip(axes, panels.items()):
        # Strictly assign baseline colors
        colours = [
            VAR_BASE if r == DEEP_BASELINE_RUN else 
            (VAR_SHALLOW if r == SHALLOW_BASELINE_RUN else SPECIES_COLOURS.get(domain, "#4C72B0")) 
            for r in df_p["run"]
        ]
        _hbar(ax, df_p["label"].tolist(), df_p["ari"].tolist(), colours)
        ax.set_xlim(0, 1.0)
        ax.set_title(domain.replace("Isolated: ", ""), fontweight="bold")
        ax.set_xlabel("ARI")

    for ax in axes[len(panels):]: ax.set_visible(False)
    fig.suptitle(f"{FAMILY_NAMES.get(family_key, family_key).replace('§ ', '')}: Taxonomy-Isolated Performance", fontsize=13, fontweight="bold", y=1.01)
    
    # The taxonomy colors are self-evident from the titles, so no explicit legend needed here, 
    # but tight_layout needs adjusting to fit the main title properly
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_species.png"))

# ============================================================================
# GLOBAL PLOTS
# ============================================================================

def plot_global_top20(peak: pd.DataFrame, global_dir: str) -> None:
    top = _filter_peak(peak).nlargest(20, PRIMARY_METRIC).copy()
    if top.empty: return
    top["Config"] = top["Run_Name"].apply(lambda r: f"[{get_family_prefix(r)}] {get_human_readable_variant(r)}")
    top = top.sort_values(PRIMARY_METRIC, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Strictly assign baseline colors
    colours = [
        VAR_BASE if r == DEEP_BASELINE_RUN else 
        (VAR_SHALLOW if r == SHALLOW_BASELINE_RUN else "#4C72B0") 
        for r in top["Run_Name"]
    ]
    
    _hbar(ax, top["Config"].tolist(), top[PRIMARY_METRIC].tolist(), colours)
    ax.set_xlabel("Adjusted Rand Index (ARI)")
    ax.set_title("Global Performance Benchmark: Top 20 Configurations (Known Distribution)", fontweight="bold")
    _save_fig(fig, os.path.join(global_dir, "global_top20_ARI.png"))

def plot_global_domain_gap(peak: pd.DataFrame, global_dir: str) -> None:
    known  = _filter_peak(peak, "Known (Mixed Species)")
    unseen = _filter_peak(peak, "Unseen (OOD)")
    merged = pd.merge(known[["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "known"}), unseen[["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "unseen"}), on="Run_Name")
    if merged.empty: return
    merged["drop"] = merged["known"] - merged["unseen"]
    merged = merged.nsmallest(15, "drop")
    merged["Config"] = merged["Run_Name"].apply(lambda r: f"[{get_family_prefix(r)}] {get_human_readable_variant(r)}")
    merged = merged.sort_values("drop", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = ["#C44E52" if d > 0 else "#55A868" for d in merged["drop"]]
    ax.barh(merged["Config"], merged["drop"], color=colours, edgecolor="white", linewidth=0.4)
    for bar, val in zip(ax.patches, merged["drop"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ARI Degradation (Known $\\rightarrow$ OOD)")
    ax.set_title("Out-of-Distribution (OOD) Robustness (Minimization of Generalization Gap)", fontweight="bold")
    _save_fig(fig, os.path.join(global_dir, "global_generalisation_gap.png"))

def plot_global_reranking(peak: pd.DataFrame, global_dir: str) -> None:
    base   = peak[(peak["Domain"] == "Known (Mixed Species)") & (~peak["Optimised"])]
    raw    = base[base["Reranking"] == "disabled"][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "raw"})
    ranked = base[base["Reranking"] == "enabled"][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "ranked"})
    merged = pd.merge(raw, ranked, on="Run_Name")
    if merged.empty: return
    merged["gain"] = merged["ranked"] - merged["raw"]
    top = merged.nlargest(15, "gain")
    top["Config"] = top["Run_Name"].apply(lambda r: f"[{get_family_prefix(r)}] {get_human_readable_variant(r)}")
    top = top.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    _hbar(ax, top["Config"].tolist(), top["gain"].tolist(), "#7B4F9E")
    ax.set_xlabel("Absolute ARI Improvement ($\\Delta$)")
    ax.set_title("Efficacy of k-Reciprocal Jaccard Reranking on Topological Embeddings", fontweight="bold")
    _save_fig(fig, os.path.join(global_dir, "global_reranking_gain.png"))

def plot_global_hdbscan(peak: pd.DataFrame, global_dir: str) -> None:
    base  = peak[(peak["Domain"] == "Known (Mixed Species)") & (peak["Reranking"] == "enabled")]
    fixed = base[~base["Optimised"]][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "fixed"})
    opt   = base[ base["Optimised"]][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "opt"})
    merged = pd.merge(fixed, opt, on="Run_Name")
    if merged.empty: return
    merged["gain"] = merged["opt"] - merged["fixed"]
    top = merged.nlargest(15, "gain")
    top["Config"] = top["Run_Name"].apply(lambda r: f"[{get_family_prefix(r)}] {get_human_readable_variant(r)}")
    top = top.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    _hbar(ax, top["Config"].tolist(), top["gain"].tolist(), "#D4A017")
    ax.set_xlabel("Absolute ARI Improvement ($\\Delta$)")
    ax.set_title("Efficacy of Empirical HDBSCAN Hyperparameter Optimization", fontweight="bold")
    _save_fig(fig, os.path.join(global_dir, "global_hdbscan_gain.png"))

def plot_global_species(peak: pd.DataFrame, global_dir: str) -> None:
    generalist_families = ["Exp0", "Exp1", "Exp2", "Exp3", "Exp4_CAE", "Exp4_Emb", "Exp5", "Exp6", "Exp7"]
    gen_runs: list[str] = []
    for fam in generalist_families: gen_runs.extend(get_target_runs(peak, fam))
    if DEEP_BASELINE_RUN not in gen_runs: gen_runs.append(DEEP_BASELINE_RUN)

    for domain in SPECIES_DOMAINS:
        rows = []
        for run in gen_runs:
            sub = peak[(peak["Run_Name"] == run) & (peak["Domain"] == domain) & (peak["Reranking"] == "enabled") & (~peak["Optimised"])]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            if pd.notna(ari):
                rows.append({"label": f"[{get_family_prefix(run)}] {get_human_readable_variant(run)}", "ari": ari, "run": run})
        if not rows: continue
        df_p = pd.DataFrame(rows).sort_values("ari", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(df_p) * 0.45)))
        
        # Strictly assign baseline colors
        colours = [
            VAR_BASE if r == DEEP_BASELINE_RUN else 
            (VAR_SHALLOW if r == SHALLOW_BASELINE_RUN else SPECIES_COLOURS.get(domain, "#4C72B0")) 
            for r in df_p["run"]
        ]
        
        _hbar(ax, df_p["label"].tolist(), df_p["ari"].tolist(), colours)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Adjusted Rand Index (ARI)")
        ax.set_title(f"Comparative Generalist Efficacy: Isolated {domain.replace('Isolated: ', '')} Taxonomy", fontweight="bold")
        _save_fig(fig, os.path.join(global_dir, f"global_species_{domain.lower().replace('isolated: ', '').replace(' ', '_')}.png"))

def plot_specialist_vs_generalist(peak: pd.DataFrame, global_dir: str) -> None:
    comparisons = [(DEEP_BASELINE_RUN, "Exp8_R23_Lynx_Only", "Isolated: Lynx"), (DEEP_BASELINE_RUN, "Exp8_R24_Salamander_Only", "Isolated: Salamander"), (DEEP_BASELINE_RUN, "Exp8_R25_Turtle_Only", "Isolated: Turtle")]
    rows = []
    for gen_run, spec_run, domain in comparisons:
        for run, role in [(gen_run, "Multi-Species Generalist (3L)"), (spec_run, "Dedicated Specialist (3L)")]:
            sub = peak[(peak["Run_Name"] == run) & (peak["Domain"] == domain) & (peak["Reranking"] == "enabled") & (~peak["Optimised"])]
            ari = sub[PRIMARY_METRIC].iloc[0] if not sub.empty else np.nan
            rows.append({"species": domain.replace("Isolated: ", ""), "role": role, "ari": ari})
    df = pd.DataFrame(rows).dropna(subset=["ari"])
    if df.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="species", y="ari", hue="role", palette={"Multi-Species Generalist (3L)": "#4C72B0", "Dedicated Specialist (3L)": "#55A868"}, ax=ax, width=0.55)
    for container in ax.containers: ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, df["ari"].max() + 0.15)
    ax.set_xlabel("")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_title("Architectural Paradigm Comparison: Generalists vs. Specialists", fontweight="bold")
    
    # Legend centered at the bottom
    ax.legend(title="Architectural Paradigm", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.8)
    _save_fig(fig, os.path.join(global_dir, "specialist_vs_generalist.png"))

# ============================================================================
# ORCHESTRATOR
# ============================================================================
def generate_all_family_plots(master: pd.DataFrame, peak: pd.DataFrame, out_dir: str) -> None:
    print("  Generating per-family plots …")
    for fam in ORDERED_FAMILIES:
        fam_dir = os.path.join(out_dir, fam)
        os.makedirs(fam_dir, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_family_generalisation(peak, fam, fam_dir)
            plot_family_learning_curves(master, peak, fam, fam_dir)
            plot_family_species(peak, fam, fam_dir)
            plot_family_reranking(peak, fam, fam_dir)
            plot_family_hdbscan(peak, fam, fam_dir)
            
            # --- NAMED LAYOUTS (Angled Text) ---
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_MAIN, 
                                      suffix="main_wide_named", max_cols=6, use_ids=False)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_APPENDIX, 
                                      suffix="appendix_wide_named", max_cols=6, use_ids=False)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_MAIN, 
                                      suffix="main_stacked_named", max_cols=3, use_ids=False)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_APPENDIX, 
                                      suffix="appendix_stacked_named", max_cols=3, use_ids=False)

            # --- NUMBERED LAYOUTS (IDs + Mapping CSVs) ---
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_MAIN, 
                                      suffix="main_wide_numbered", max_cols=6, use_ids=True)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_APPENDIX, 
                                      suffix="appendix_wide_numbered", max_cols=6, use_ids=True)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_MAIN, 
                                      suffix="main_stacked_numbered", max_cols=3, use_ids=True)
            plot_compact_metrics_grid(peak, fam, fam_dir, panel_list=METRIC_PANELS_APPENDIX, 
                                      suffix="appendix_stacked_numbered", max_cols=3, use_ids=True)

def generate_all_global_plots(master: pd.DataFrame, peak: pd.DataFrame, global_dir: str) -> None:
    print("  Generating global plots …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_global_top20(peak, global_dir)
        plot_global_domain_gap(peak, global_dir)
        plot_global_reranking(peak, global_dir)
        plot_global_hdbscan(peak, global_dir)
        plot_global_species(peak, global_dir)
        plot_specialist_vs_generalist(peak, global_dir)


# ============================================================================
#VARIATION PANELS  (one PNG per domain, like images 1 & 2 in reference)
# ============================================================================

#: Canonical domain display order for panel columns
PANEL_DOMAINS = [
    # --- Row 1 (Chunk 1): Global & Mixed Distributions ---
    ("Known (Mixed Species)",  "known"),
    ("Mixed (Known + Unseen)", "global_mix"),
    ("Unseen (OOD)",           "ood"),
    
    # --- Row 2 (Chunk 2): Isolated Species Taxonomy ---
    ("Isolated: Lynx",         "lynx"),
    ("Isolated: Salamander",   "salamander"),
    ("Isolated: Turtle",       "turtle"),
]

def _domain_short(domain: str) -> str:
    """Shortens domain names with newlines to prevent overlap in narrow columns."""
    mapping = {
        "Known (Mixed Species)": "Known\nMixed",
        "Mixed (Known + Unseen)": "Global\nMixed",
        "Isolated: Lynx": "Lynx",
        "Isolated: Salamander": "Salamander",
        "Isolated: Turtle": "Turtle",
        "Unseen (OOD)": "OOD"
    }
    return mapping.get(domain, domain)

# --- Configuration for Main Paper (Floor vs Ceiling) ---
METRIC_PANELS_MAIN = [
    ("Baseline ARI", "Native GNN | Static HDBScan", "disabled", False), 
    ("Baseline ARI", "Reranked | Opt HDBScan", "enabled",  True),
]

# --- Configuration for Appendix (Full Ablation) ---
METRIC_PANELS_APPENDIX = [
    ("Baseline ARI", "No Reranking | Fixed HDB", "disabled", False),
    ("Baseline ARI", "Reranked | Fixed HDB",     "enabled",  False),
    ("Baseline ARI", "Reranked | Opt HDB",       "enabled",  True),
]

def plot_compact_metrics_grid(peak: pd.DataFrame, family_key: str, fam_dir: str, 
                              panel_list: list, suffix: str = "compact", 
                              max_cols: int = 6, use_ids: bool = False) -> None:
    """Ultimate Grid: Uniform dynamic Y-scale, missing-bar safety, and optimized spacing."""
    runs = get_target_runs(peak, family_key)
    if not runs: return

    # --- BASELINE OVERRIDE & DYNAMIC SORTING ---
    baseline_run, baseline_label = choose_baseline(list(runs))
    
    if "Exp0" in family_key:
        all_temp = list(runs)
        if baseline_run not in all_temp: all_temp.append(baseline_run)
        if SHALLOW_BASELINE_RUN not in all_temp: all_temp.append(SHALLOW_BASELINE_RUN)
        def get_exp0_rank(r):
            if r == SHALLOW_BASELINE_RUN: return 1
            if r == baseline_run: return 2
            return 99
        all_runs = sorted(all_temp, key=get_exp0_rank)
        
    elif "Exp3" in family_key:
        baseline_run = SHALLOW_BASELINE_RUN
        baseline_label = "Baseline (CAE+Pos+Color)"
        valid_runs = [r for r in runs if r not in (DEEP_BASELINE_RUN, "Exp3_R10B_Baseline")]
        if SHALLOW_BASELINE_RUN not in valid_runs: valid_runs.append(SHALLOW_BASELINE_RUN)
        
        exp3_sort_map = {
            "Exp3_R08_Color": 1, "Exp3_R10_PureTexture": 2, "Exp3_R10A_Texture_Pos": 3,
            "Exp3_R09_ClassicVision": 4, SHALLOW_BASELINE_RUN: 5, "Exp3_R10D_Baseline_Plus_Shape": 6,
            "Exp3_R10C_Baseline_Plus_HOG": 7, "Exp3_R11_all features": 8
        }
        def get_exp3_rank(r):
            for key, rank in exp3_sort_map.items():
                if key in r: return rank
            if r == SHALLOW_BASELINE_RUN: return 5
            return 99 
        all_runs = sorted(valid_runs, key=get_exp3_rank)
        
    elif "Exp4_Emb" in family_key:
        all_temp = list(runs)
        if baseline_run not in all_temp: all_temp.append(baseline_run)
        if SHALLOW_BASELINE_RUN not in all_temp: all_temp.append(SHALLOW_BASELINE_RUN)
        def get_emb_rank(r):
            if r == baseline_run: return 512
            if r == SHALLOW_BASELINE_RUN: return 511
            if "Dim32" in r: return 32
            if "Dim64" in r: return 64
            if "Dim128" in r: return 128
            if "Dim256" in r: return 256
            if "Dim1024" in r: return 1024
            return 9999
        all_runs = sorted(all_temp, key=get_emb_rank)

    elif "Exp4_CAE" in family_key:
        all_temp = list(runs)
        if baseline_run not in all_temp: all_temp.append(baseline_run)
        if SHALLOW_BASELINE_RUN not in all_temp: all_temp.append(SHALLOW_BASELINE_RUN)
        def get_cae_rank(r):
            if r == baseline_run: return 90
            if r == SHALLOW_BASELINE_RUN: return 89
            if "Dim32" in r: return 32
            if "Dim64" in r: return 64
            if "Dim128" in r: return 128
            return 9999
        all_runs = sorted(all_temp, key=get_cae_rank)

    elif "Exp5" in family_key:
        # === EXP 5 CUSTOM SORTING RULE ===
        all_temp = list(runs)
        if baseline_run not in all_temp: all_temp.append(baseline_run)
        if SHALLOW_BASELINE_RUN not in all_temp: all_temp.append(SHALLOW_BASELINE_RUN)
        def get_exp5_rank(r):
            if r == SHALLOW_BASELINE_RUN: return 1
            if "Pool_Mean" in r or "R17" in r: return 2
            if "Pool_Max" in r or "R18" in r: return 3
            if r == baseline_run: return 4
            if "No_OrthoLoss" in r or "R19" in r: return 5
            return 99
        all_runs = sorted(all_temp, key=get_exp5_rank)
        
    elif "Exp6" in family_key:
        # === EXP 6 CUSTOM SORTING RULE ===
        baseline_run = SHALLOW_BASELINE_RUN
        baseline_label = "Baseline (1L 240SP)"
        valid_runs = [r for r in runs if r not in (DEEP_BASELINE_RUN, "Exp3_R10B_Baseline")]
        if SHALLOW_BASELINE_RUN not in valid_runs: valid_runs.append(SHALLOW_BASELINE_RUN)
        
        def get_exp6_rank(r):
            if "50SP" in r or "R20" in r: return 1
            if r == SHALLOW_BASELINE_RUN: return 2
            if "500SP" in r or "R21" in r: return 3
            return 99
        all_runs = sorted(valid_runs, key=get_exp6_rank)
    
    elif "Exp7" in family_key:
        baseline_run = SHALLOW_BASELINE_RUN
        baseline_label = "Baseline (1L Spatial)"
        valid_runs = [r for r in runs if r != DEEP_BASELINE_RUN]
        if SHALLOW_BASELINE_RUN not in valid_runs: valid_runs.append(SHALLOW_BASELINE_RUN)
        
        def get_exp7_rank(r):
            if r == SHALLOW_BASELINE_RUN: return 2
            return 1
        all_runs = sorted(valid_runs, key=get_exp7_rank)
    elif "Exp8" in family_key:
        baseline_run = DEEP_BASELINE_RUN
        baseline_label = "Generalist (3L)"
        
        # Explicit order: 3L Gen, 1L Gen, Lynx 3L/1L, Salamander 3L/1L, Turtle 3L/1L
        target_order = [
            DEEP_BASELINE_RUN, SHALLOW_BASELINE_RUN,
            "Exp8_R23_Lynx_Only", "Exp8_R26_Lynx_Probe_1L",
            "Exp8_R24_Salamander_Only", "Exp8_R28_Salamander_Probe_1L",
            "Exp8_R25_Turtle_Only", "Exp8_R27_Turtle_Probe_1L"
        ]
        
        all_runs = [r for r in target_order if r in runs or r in [DEEP_BASELINE_RUN, SHALLOW_BASELINE_RUN]]
    elif "Exp9" in family_key:
        baseline_run = DEEP_BASELINE_RUN
        baseline_label = "Generalist (3L)"
        
        target_order = [
            DEEP_BASELINE_RUN,
            "Exp9_H2_Salamander_Micro500SP",
            "Exp9_H3_Salamander_Hybrid_Micro",
            "Exp9_H4_Turtle_PureTexture",
            "Exp9_H5_Turtle_ShapeAndTexture"
        ]
        all_runs = [r for r in target_order if r in runs or r == DEEP_BASELINE_RUN]

    elif "Exp10" in family_key:
        baseline_run = DEEP_BASELINE_RUN
        baseline_label = "Baseline (3L Spatial)"
        
        # Order: Drop 1 (Modality), Drop 2 (Graph/Node+Edge), Drop 3 (All), Baseline
        target_order = [
            "Exp10_R36_No_ModalityDrop", 
            "Exp10_R37_No_GraphDrop",    
            "Exp10_R35_No_Dropouts",     
            DEEP_BASELINE_RUN            
        ]
        all_runs = [r for r in target_order if r in runs or r == DEEP_BASELINE_RUN]
    elif "Exp11" in family_key:
        baseline_run = DEEP_BASELINE_RUN
        baseline_label = "Baseline (Original Backgrounds)"
        
        # Force the exact visual order from left to right
        def get_exp11_rank(r):
            if r == DEEP_BASELINE_RUN: return 1
            if "R38" in r: return 2   # 1st: Turtle
            if "R39" in r: return 3   # 2nd: Salamander
            if "R23" in r: return 4   # 3rd: Lynx
            if "R41" in r: return 5   # 4th: All Species
            return 99
            
        all_temp = list(runs)
        if baseline_run not in all_temp: all_temp.append(baseline_run)
        all_runs = sorted(all_temp, key=get_exp11_rank)
    # ------------------------------------

    else:
        all_runs = sorted([r for r in runs if r != baseline_run])
        all_runs.append(baseline_run)
    n_runs = len(all_runs)
    run_to_id = {run: str(i+1) for i, run in enumerate(all_runs)}
    
    n_domains = len(PANEL_DOMAINS)
    n_cols = min(n_domains, max_cols)
    domain_chunks = [PANEL_DOMAINS[i:i + n_cols] for i in range(0, n_domains, n_cols)]
    total_rows = len(domain_chunks) * len(panel_list)

    col_width = 1.2 if family_key in ["Exp0", "Exp7", "Exp8", "Exp9", "Exp10", "Exp11"] else (n_runs * 0.2)
    row_height = 1.2
    
    fig, axes = plt.subplots(total_rows, n_cols, 
                             figsize=(n_cols * col_width, total_rows * row_height), 
                             squeeze=False)

    absolute_max = 0.05
    for r_stage, (metric_col, metric_label, rerank_val, opt_val) in enumerate(panel_list):
        mask = (peak["Run_Name"].isin(all_runs) & (peak["Reranking"] == rerank_val) & (peak["Optimised"] == opt_val))
        stage_max = peak[mask][metric_col].max()
        if pd.notna(stage_max) and stage_max > absolute_max: absolute_max = stage_max
            
    safe_max = absolute_max
    global_y_lim = safe_max * 1.25 

    ax_flat = axes.flatten()
    curr_ax_idx = 0

    for chunk_idx, chunk in enumerate(domain_chunks):
        for r_stage, (metric_col, metric_label, rerank_val, opt_val) in enumerate(panel_list):
            physical_row = chunk_idx * len(panel_list) + r_stage
            for c_idx, (domain, domain_tag) in enumerate(chunk):
                ax = ax_flat[curr_ax_idx]
                v_rows = peak[(peak["Run_Name"].isin(all_runs)) & (peak["Domain"] == domain) &
                              (peak["Reranking"] == rerank_val) & (peak["Optimised"] == opt_val)]
                
                rows = []
                for run in all_runs:
                    val_row = v_rows[v_rows["Run_Name"] == run]
                    val = val_row[metric_col].iloc[0] if not val_row.empty and pd.notna(val_row[metric_col].iloc[0]) else 0.0
                    
                    if use_ids:
                        display_label = run_to_id[run]
                    else:
                        if "Exp0" in family_key:
                            if run == baseline_run: display_label = "Baseline (3L Spatial)"
                            elif run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L Spatial)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp1" in family_key:
                            if run == "Exp1_R01_Edges_Attn": display_label = "Edges Attn 3L"
                            elif run == "Exp1_R02_Edges_Hybrid": display_label = "Edges Hybrid 3L"
                            elif run == "Exp1_R02b_Edges_Hybrid_2L": display_label = "Edges Hybrid 2L"
                            elif run == baseline_run: display_label = "Baseline (3L Spatial)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp3" in family_key:
                            if run == SHALLOW_BASELINE_RUN: display_label = "Baseline (CAE+Pos+Color)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp4_Emb" in family_key:
                            if run == baseline_run: display_label = "Baseline (3L Dim 512)"
                            elif run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L Dim 512)"
                            elif "Dim32" in run: display_label = "1L Dim 32"
                            elif "Dim64" in run: display_label = "1L Dim 64"
                            elif "Dim128" in run: display_label = "1L Dim 128"
                            elif "Dim256" in run: display_label = "1L Dim 256"
                            elif "Dim1024" in run: display_label = "1L Dim 1024"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp4_CAE" in family_key:
                            if run == baseline_run: display_label = "Baseline (3L Dim 90)"
                            elif run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L Dim 90)"
                            elif "Dim32" in run: display_label = "1L Dim 32"
                            elif "Dim64" in run: display_label = "1L Dim 64"
                            elif "Dim128" in run: display_label = "1L Dim 128"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp5" in family_key:
                            if run == baseline_run: display_label = "Baseline (3L GeM)"
                            elif run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L GeM)"
                            elif "R19" in run: display_label = "No Orthogonal Loss"
                            elif "R20" in run: display_label = "Shared Task Weights"
                            elif "R21" in run: display_label = "No Stochastic Dropout"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp6" in family_key:
                            if run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L 240SP)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        
                        elif "Exp7" in family_key:
                            if run == SHALLOW_BASELINE_RUN: display_label = "Baseline (1L Spatial)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp8" in family_key:
                            if run == DEEP_BASELINE_RUN: display_label = "Generalist (3L)"
                            elif run == SHALLOW_BASELINE_RUN: display_label = "Generalist (1L)"
                            elif "R23" in run: display_label = "Specialist Lynx (3L)"
                            elif "R26" in run: display_label = "Specialist Lynx (1L)"
                            elif "R24" in run: display_label = "Specialist Salamander (3L)"
                            elif "R28" in run: display_label = "Specialist Salamander (1L)"
                            elif "R25" in run: display_label = "Specialist Turtle (3L)"
                            elif "R27" in run: display_label = "Specialist Turtle (1L)"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp9" in family_key:
                            if run == DEEP_BASELINE_RUN: display_label = "Generalist (3L)"
                            elif "H2" in run: display_label = "H2: Salamander Micro"
                            elif "H3" in run: display_label = "H3: Salamander Hybrid+Micro"
                            elif "H4" in run: display_label = "H4: Turtle Pure Texture"
                            elif "H5" in run: display_label = "H5: Turtle Shape+Texture"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        
                        elif "Exp10" in family_key:
                            if run == DEEP_BASELINE_RUN: display_label = "Baseline (3L Spatial)"
                            elif "R36" in run: display_label = "No Modality Dropout"
                            elif "R37" in run: display_label = "No Graph Dropout"
                            elif "R35" in run: display_label = "No Dropouts"
                            else: display_label = _consistent_label(run, baseline_run, baseline_label)
                        elif "Exp11" in family_key:
                            baseline_run = DEEP_BASELINE_RUN
                            baseline_label = "Baseline (Original Backgrounds)"
                            target_order = [
                                "Exp8_R23_Lynx_Only",
                                "Exp11_R38_Turtle_Cutouts",
                                "Exp11_R39_Salamander_Cutouts",
                                "Exp11_R41_All_Species_Cutouts",
                                DEEP_BASELINE_RUN
                            ]
                            all_runs = [r for r in target_order if r in runs or r in [DEEP_BASELINE_RUN, "Exp8_R23_Lynx_Only"]]
                        else:
                            display_label = _consistent_label(run, baseline_run, baseline_label)

                    # STRICT COLOR ASSIGNMENT TAGS
                    rows.append({
                        "label": display_label, 
                        "value": val, 
                        "is_deep_base": run == DEEP_BASELINE_RUN,
                        "is_shallow_base": run == SHALLOW_BASELINE_RUN
                    })

                if rows:
                    df = pd.DataFrame(rows)
                    max_val = df["value"].max()
                    colours = []
                    for _, row in df.iterrows():
                        # STRICT COLOR ASSIGNMENT LOGIC
                        if row["is_deep_base"]: colours.append(VAR_BASE)
                        elif row["is_shallow_base"]: colours.append(VAR_SHALLOW)
                        elif row["value"] == max_val: colours.append(SPECIES_COLOURS.get(domain, "#4C72B0"))
                        else: colours.append(mcolors.to_rgba(SPECIES_COLOURS.get(domain, "#4C72B0"), 0.35))

                    bars = ax.bar(range(len(df)), df["value"], color=colours, width=0.65, edgecolor="white", linewidth=0.3)
                    ax.set_ylim(0, global_y_lim)
                    y_ticks = np.round(np.linspace(0, global_y_lim, num=4), 1)
                    ax.set_yticks(y_ticks)
                    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

                    for i, (bar, val) in enumerate(zip(bars, df["value"])):
                        if val > 0.0:
                            if "Exp8" in family_key:
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (safe_max * 0.03),
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=5.0, 
                                    fontweight=("bold" if val == max_val else "normal"), rotation=55)
                            else:
                                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (safe_max * 0.03),
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=5.0, 
                                    fontweight=("bold" if val == max_val else "normal"), rotation=45)

                    if r_stage == 0:
                        ax.set_title(f"{_domain_short(domain)}", fontweight="bold", fontsize=6.0, pad=10)
                    
                    if c_idx == 0:
                        ax.set_ylabel(metric_label.replace(" | ", "\n"), fontsize=8, labelpad=2)
                        ax.tick_params(axis='y', labelsize=7.0)
                    else:
                        ax.set_yticklabels([]); ax.tick_params(axis='y', length=0)
                    
                    ax.set_xticks(range(len(df)))
                    if physical_row == total_rows - 1:
                        ax.set_xticklabels(df["label"], rotation=55 if not use_ids else 0, ha="right" if not use_ids else "center", fontsize=5.5 if not use_ids else 7.5)
                    else:
                        ax.set_xticklabels([]); ax.tick_params(axis='x', length=0)
                else:
                    ax.axis('off')
                curr_ax_idx += 1
            while curr_ax_idx % n_cols != 0:
                ax_flat[curr_ax_idx].axis('off')
                curr_ax_idx += 1

    bottom_margin = 0.10 if use_ids else 0.25
    plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=bottom_margin, 
                        wspace=(0.08 if family_key in ["Exp0", "Exp7", "Exp8", "Exp9", "Exp10", "Exp11"] else 0.05), 
                        hspace=0.35)
    _save_fig(fig, os.path.join(fam_dir, f"{family_key}_metrics_grid_{suffix}.png"))

    if use_ids:
        mapping_data = []
        for run in all_runs:
            c_id = run_to_id[run]
            if "Exp0" in family_key:
                if run == baseline_run: c_label = "Baseline (3L Spatial)"
                elif run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L Spatial)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp1" in family_key:
                if run == "Exp1_R01_Edges_Attn": c_label = "Edges Attn 3L"
                elif run == "Exp1_R02_Edges_Hybrid": c_label = "Edges Hybrid 3L"
                elif run == "Exp1_R02b_Edges_Hybrid_2L": c_label = "Edges Hybrid 2L"
                elif run == baseline_run: c_label = "Baseline (3L Spatial)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp3" in family_key:
                if run == SHALLOW_BASELINE_RUN: c_label = "Baseline (CAE+Pos+Color)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp4_Emb" in family_key:
                if run == baseline_run: c_label = "Baseline (3L Dim 512)"
                elif run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L Dim 512)"
                elif "Dim32" in run: c_label = "1L Dim 32"
                elif "Dim64" in run: c_label = "1L Dim 64"
                elif "Dim128" in run: c_label = "1L Dim 128"
                elif "Dim256" in run: c_label = "1L Dim 256"
                elif "Dim1024" in run: c_label = "1L Dim 1024"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp4_CAE" in family_key:
                if run == baseline_run: c_label = "Baseline (3L Dim 90)"
                elif run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L Dim 90)"
                elif "Dim32" in run: c_label = "1L Dim 32"
                elif "Dim64" in run: c_label = "1L Dim 64"
                elif "Dim128" in run: c_label = "1L Dim 128"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp5" in family_key:
                if run == baseline_run: c_label = "Baseline (3L GeM)"
                elif run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L GeM)"
                elif "R19" in run: c_label = "No Orthogonal Loss"
                elif "R20" in run: c_label = "Shared Task Weights"
                elif "R21" in run: c_label = "No Stochastic Dropout"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp6" in family_key:
                if run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L 240SP)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp7" in family_key:
                if run == SHALLOW_BASELINE_RUN: c_label = "Baseline (1L Spatial)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
                
            # --- UPDATED TO MATCH PAPER NARRATIVE ---
            elif "Exp8" in family_key:
                # Exp 8 is now Regularization Approaches
                if run == DEEP_BASELINE_RUN: c_label = "Baseline (All Dropouts Enabled)"
                elif "R37" in run: c_label = "Modality Dropout Only (No Graph)"
                elif "R36" in run: c_label = "Structural Dropout Only (No Modality)"
                elif "R35" in run: c_label = "Unregularized (No Dropouts)"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
                
            elif "Exp9" in family_key:
                # Exp 9 is now Extended Training Data
                if run == SHALLOW_BASELINE_RUN: c_label = "Baseline (Standard Data)"
                elif "universal" in run.lower() or "extended" in run.lower(): c_label = "Extended Multi-Species Data"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
                
            elif "Exp11" in family_key:
                # FIXED: Changed display_label to c_label to prevent python NameError crash
                if run == DEEP_BASELINE_RUN: c_label = "Baseline (Original Backgrounds)"
                elif "R23" in run: c_label = "Lynx Specialist (Original BG)"
                elif "R38" in run: c_label = "Turtle Cutouts Only"
                elif "R39" in run: c_label = "Salamander Cutouts Only"
                elif "R41" in run: c_label = "All Species Cutouts"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            elif "Exp11" in family_key:
                if run == DEEP_BASELINE_RUN: c_label = "Baseline (Original Backgrounds)"
                elif "R38" in run: c_label = "Turtle Cutouts Only"
                elif "R39" in run: c_label = "Salamander Cutouts Only"
                elif "R23" in run: c_label = "Lynx Cutouts Only"  
                elif "R41" in run: c_label = "All Species Cutouts"
                else: c_label = _consistent_label(run, baseline_run, baseline_label)
            else:
                c_label = _consistent_label(run, baseline_run, baseline_label)
                
            mapping_data.append({"Bar ID": c_id, "LaTeX Label": c_label, "Run Folder": run})
            
        map_df = pd.DataFrame(mapping_data)
        map_df.to_csv(os.path.join(fam_dir, f"{family_key}_mapping_{suffix}.csv"), index=False)
        tex_fpath = os.path.join(fam_dir, f"{family_key}_mapping_{suffix}.tex")
        tex_lines = ["\\begin{table}[htbp]", "  \\centering", f"  \\caption{{Ablation Mapping for {family_key.replace('_', '\\\\_')}}}",
                     f"  \\label{{tab:{family_key.lower()}_mapping}}", "  \\begin{tabular}{c l}", "    \\toprule", "    \\textbf{ID} & \\textbf{Configuration} \\\\", "    \\midrule"]
        for item in mapping_data:
            tex_lines.append(f"    {item['Bar ID']} & {item['LaTeX Label'].replace('%', '\\\\%').replace('_', '\\\\_')} \\\\")
        tex_lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
        with open(tex_fpath, "w", encoding="utf-8") as f: f.write("\n".join(tex_lines))

def _consistent_label(run: str, baseline_run: str, baseline_label: str) -> str:
    """Ensures the baseline model always gets its unified label."""
    if run == baseline_run:
        return baseline_label
    return get_human_readable_variant(run)