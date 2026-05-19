import os
import numpy as np
import pandas as pd

from config import (
    PRIMARY_METRIC, SECONDARY_METRIC,
    ALL_RETRIEVAL_METRICS, ALL_CLUSTER_METRICS, ALL_OPENSET_METRICS,
    DEEP_BASELINE_RUN, SHALLOW_BASELINE_RUN, ORDERED_FAMILIES, FAMILY_NAMES,
    get_target_runs, get_human_readable_variant, choose_baseline, is_shallow_probe,
)

SPECIES_DOMAINS = ["Isolated: Lynx", "Isolated: Salamander", "Isolated: Turtle"]


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================

def _lookup(peak: pd.DataFrame, run: str, domain: str,
            rerank: str, opt: bool,
            metric: str = PRIMARY_METRIC) -> float:
    mask = (
        (peak["Run_Name"] == run) &
        (peak["Domain"]   == domain) &
        (peak["Reranking"] == rerank) &
        (peak["Optimised"] == opt)
    )
    vals = peak.loc[mask, metric]
    return round(float(vals.iloc[0]), 4) if not vals.empty else np.nan


def _fmt(v) -> str:
    return f"{v:.4f}" if pd.notna(v) else "—"


def _save(df: pd.DataFrame, path_no_ext: str) -> None:
    """Write both CSV and Markdown versions."""
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_markdown(path_no_ext + ".md", index=False, floatfmt=".4f")
    except ImportError:
        pass   # tabulate not installed — CSV is enough


# ============================================================================
# PER-FAMILY TABLES
# ============================================================================

def _family_performance_table(peak: pd.DataFrame, family_key: str) -> pd.DataFrame | None:
    runs = get_target_runs(peak, family_key)
    if not runs:
        return None

    baseline_run, baseline_label = choose_baseline(runs)

    all_runs = list(runs)
    if baseline_run not in all_runs:
        all_runs.append(baseline_run)

    records = []
    for run in all_runs:
        row_type = (
            "Baseline"     if run == baseline_run else
            "Shallow probe" if is_shallow_probe(run) else
            "Standard"
        )
        label = baseline_label if run == baseline_run else get_human_readable_variant(run)

        records.append({
            "Configuration": label,
            "Type":          row_type,
            "ARI — Known":   _fmt(_lookup(peak, run, "Known (Mixed Species)", "enabled", False)),
            "ARI — OOD":     _fmt(_lookup(peak, run, "Unseen (OOD)",          "enabled", False)),
            "ARI — Raw":     _fmt(_lookup(peak, run, "Known (Mixed Species)", "disabled", False)),
            "ARI — Opt":     _fmt(_lookup(peak, run, "Known (Mixed Species)", "enabled",  True)),
            "mAP (%)":       _fmt(_lookup(peak, run, "Known (Mixed Species)", "enabled", False, "mAP (%)")),
            "Rank-1 (%)":    _fmt(_lookup(peak, run, "Known (Mixed Species)", "enabled", False, "Rank-1 (%)")),
            "_sort":         _lookup(peak, run, "Known (Mixed Species)", "enabled", False),
        })

    df = (pd.DataFrame(records)
            .sort_values("_sort", ascending=False, na_position="last")
            .drop(columns=["_sort"])
            .reset_index(drop=True))
    return df


def _family_species_table(peak: pd.DataFrame, family_key: str) -> pd.DataFrame | None:
    runs = get_target_runs(peak, family_key)
    if not runs:
        return None

    baseline_run, baseline_label = choose_baseline(runs)
    all_runs = list(runs) + ([baseline_run] if baseline_run not in runs else [])

    species_data = peak[
        peak["Domain"].isin(SPECIES_DOMAINS) &
        peak["Run_Name"].isin(all_runs) &
        (peak["Reranking"] == "enabled") &
        (~peak["Optimised"])
    ]
    if species_data.empty:
        return None

    records = []
    for run in all_runs:
        row_type = "Baseline" if run == baseline_run else ("Shallow probe" if is_shallow_probe(run) else "Standard")
        label = baseline_label if run == baseline_run else get_human_readable_variant(run)
        records.append({
            "Configuration":    label,
            "Type":             row_type,
            "ARI — Lynx":       _fmt(_lookup(peak, run, "Isolated: Lynx",       "enabled", False)),
            "ARI — Salamander": _fmt(_lookup(peak, run, "Isolated: Salamander", "enabled", False)),
            "ARI — Turtle":     _fmt(_lookup(peak, run, "Isolated: Turtle",     "enabled", False)),
            "_sort":            _lookup(peak, run, "Isolated: Lynx",             "enabled", False),
        })

    df = (pd.DataFrame(records)
            .sort_values("_sort", ascending=False, na_position="last")
            .drop(columns=["_sort"])
            .reset_index(drop=True))
    return df if (df[["ARI — Lynx", "ARI — Salamander", "ARI — Turtle"]]
                  .replace("—", np.nan).notna().any().any()) else None


def generate_latex_table_rows(fam_dir: str, family_key: str):
    """Reads the CSVs and generates a .tex snippet with the exact LaTeX rows."""
    perf_csv = os.path.join(fam_dir, f"{family_key}_performance.csv")
    spec_csv = os.path.join(fam_dir, f"{family_key}_species.csv")
    out_tex  = os.path.join(fam_dir, f"{family_key}_table_rows.tex")

    if not os.path.exists(perf_csv) or not os.path.exists(spec_csv):
        return

    df_perf = pd.read_csv(perf_csv)
    df_spec = pd.read_csv(spec_csv)
    df_merged = pd.merge(df_perf, df_spec, on=["Configuration", "Type"], how="left")

    def _safe_tex(val, is_pct=False):
        if pd.isna(val) or val == "—": return "—"
        v_str = str(val)
        if is_pct: return v_str + "\\%" if "%" not in v_str else v_str.replace("%", "\\%")
        return v_str

    lines = []
    for _, row in df_merged.iterrows():
        line = (f"{row['Configuration']} & {_safe_tex(row.get('ARI — Known'))} & {_safe_tex(row.get('ARI — OOD'))} & "
                f"{_safe_tex(row.get('ARI — Raw'))} & {_safe_tex(row.get('ARI — Opt'))} & "
                f"{_safe_tex(row.get('mAP (%)'), True)} & {_safe_tex(row.get('Rank-1 (%)'), True)} & "
                f"{_safe_tex(row.get('ARI — Lynx'))} & {_safe_tex(row.get('ARI — Salamander'))} & {_safe_tex(row.get('ARI — Turtle'))} \\\\")
        lines.append(line)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_all_family_tables(peak: pd.DataFrame, out_dir: str) -> None:
    print("  Generating per-family tables …")
    for fam in ORDERED_FAMILIES:
        fam_dir = os.path.join(out_dir, fam)
        os.makedirs(fam_dir, exist_ok=True)

        tbl = _family_performance_table(peak, fam)
        if tbl is not None and not tbl.empty:
            _save(tbl, os.path.join(fam_dir, f"{fam}_performance"))

        spec_tbl = _family_species_table(peak, fam)
        if spec_tbl is not None and not spec_tbl.empty:
            _save(spec_tbl, os.path.join(fam_dir, f"{fam}_species"))
            
        generate_latex_table_rows(fam_dir, fam)


# ============================================================================
# GLOBAL TABLES
# ============================================================================

def global_top_n(peak: pd.DataFrame, global_dir: str, n: int = 20) -> None:
    base = peak[(peak["Domain"] == "Known (Mixed Species)") & (peak["Reranking"] == "enabled") & (~peak["Optimised"])].nlargest(n, PRIMARY_METRIC).copy()
    if base.empty: return
    base["Configuration"] = base["Run_Name"].apply(get_human_readable_variant)
    base["Type"] = base["Run_Name"].apply(lambda r: "Baseline" if r == DEEP_BASELINE_RUN else ("Shallow probe" if is_shallow_probe(r) else "Standard"))
    out = base[["Configuration", "Type", PRIMARY_METRIC, SECONDARY_METRIC, "Rank-1 (%)"]].copy()
    out.columns = ["Configuration", "Type", "ARI", "mAP (%)", "Rank-1 (%)"]
    _save(out, os.path.join(global_dir, "global_top20_ARI"))

def global_generalisation_gap(peak: pd.DataFrame, global_dir: str) -> None:
    known  = peak[(peak["Domain"] == "Known (Mixed Species)") & (peak["Reranking"] == "enabled") & (~peak["Optimised"])]
    unseen = peak[(peak["Domain"] == "Unseen (OOD)") & (peak["Reranking"] == "enabled") & (~peak["Optimised"])]
    merged = pd.merge(known[["Run_Name", PRIMARY_METRIC]], unseen[["Run_Name", PRIMARY_METRIC]], on="Run_Name", suffixes=("_known", "_unseen"))
    if merged.empty: return
    merged["ARI Drop"] = (merged[f"{PRIMARY_METRIC}_known"] - merged[f"{PRIMARY_METRIC}_unseen"])
    merged["Configuration"] = merged["Run_Name"].apply(get_human_readable_variant)
    merged["Type"] = merged["Run_Name"].apply(lambda r: "Shallow probe" if is_shallow_probe(r) else "Standard")
    out = merged[["Configuration", "Type", f"{PRIMARY_METRIC}_known", f"{PRIMARY_METRIC}_unseen", "ARI Drop"]].sort_values("ARI Drop")
    out.columns = ["Configuration", "Type", "ARI (Known)", "ARI (OOD)", "ARI Drop"]
    _save(out, os.path.join(global_dir, "generalisation_gap"))

def global_reranking_impact(peak: pd.DataFrame, global_dir: str) -> None:
    base = peak[(peak["Domain"] == "Known (Mixed Species)") & (~peak["Optimised"])]
    raw = base[base["Reranking"] == "disabled"][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "Raw"})
    ranked = base[base["Reranking"] == "enabled"][["Run_Name", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "Reranked"})
    merged = pd.merge(raw, ranked, on="Run_Name")
    if merged.empty: return
    merged["Gain"] = merged["Reranked"] - merged["Raw"]
    merged["Configuration"] = merged["Run_Name"].apply(get_human_readable_variant)
    merged["Type"] = merged["Run_Name"].apply(lambda r: "Shallow probe" if is_shallow_probe(r) else "Standard")
    out = merged[["Configuration", "Type", "Raw", "Reranked", "Gain"]].sort_values("Gain", ascending=False)
    out.columns = ["Configuration", "Type", "ARI (no rerank)", "ARI (reranked)", "Gain"]
    _save(out, os.path.join(global_dir, "reranking_impact"))

def global_hdbscan_impact(peak: pd.DataFrame, global_dir: str) -> None:
    base   = peak[(peak["Domain"] == "Known (Mixed Species)") & (peak["Reranking"] == "enabled")]
    fixed  = base[~base["Optimised"]][["Run_Name", PRIMARY_METRIC, "Opt Eps", "Opt MinCls"]].rename(columns={PRIMARY_METRIC: "Fixed ARI"})
    opt    = base[ base["Optimised"]][["Run_Name", PRIMARY_METRIC, "Opt Eps", "Opt MinCls"]].rename(columns={PRIMARY_METRIC: "Opt ARI", "Opt Eps": "Best Eps", "Opt MinCls": "Best MinCls"})
    merged = pd.merge(fixed[["Run_Name", "Fixed ARI"]], opt, on="Run_Name")
    if merged.empty: return
    merged["Gain"] = merged["Opt ARI"] - merged["Fixed ARI"]
    merged["Configuration"] = merged["Run_Name"].apply(get_human_readable_variant)
    out = merged[["Configuration", "Fixed ARI", "Opt ARI", "Gain", "Best Eps", "Best MinCls"]].sort_values("Gain", ascending=False)
    _save(out, os.path.join(global_dir, "hdbscan_optimization_impact"))

def global_species_comparison(peak: pd.DataFrame, global_dir: str) -> None:
    generalist_families = ["Exp0", "Exp1", "Exp2", "Exp3", "Exp4_CAE", "Exp4_Emb", "Exp5", "Exp6", "Exp7"]
    gen_runs: list[str] = []
    for fam in generalist_families: gen_runs.extend(get_target_runs(peak, fam))
    if DEEP_BASELINE_RUN not in gen_runs: gen_runs.append(DEEP_BASELINE_RUN)
    subset = peak[peak["Run_Name"].isin(gen_runs) & peak["Domain"].isin(SPECIES_DOMAINS + ["Known (Mixed Species)"]) & (peak["Reranking"] == "enabled") & (~peak["Optimised"])].copy()
    if subset.empty: return
    records = []
    for run in gen_runs:
        records.append({
            "Configuration":    get_human_readable_variant(run),
            "Family":           _family_of(run),
            "ARI — Known":      _fmt(_lookup(peak, run, "Known (Mixed Species)", "enabled", False)),
            "ARI — Lynx":       _fmt(_lookup(peak, run, "Isolated: Lynx",        "enabled", False)),
            "ARI — Salamander": _fmt(_lookup(peak, run, "Isolated: Salamander",  "enabled", False)),
            "ARI — Turtle":     _fmt(_lookup(peak, run, "Isolated: Turtle",      "enabled", False)),
            "_sort":            _lookup(peak, run, "Known (Mixed Species)",      "enabled", False),
        })
    df = (pd.DataFrame(records).sort_values("_sort", ascending=False, na_position="last").drop(columns=["_sort"]).reset_index(drop=True))
    _save(df, os.path.join(global_dir, "generalist_species_breakdown"))

def _family_of(run_name: str) -> str:
    from config import get_family_prefix
    return get_family_prefix(run_name)

def global_open_set_summary(peak: pd.DataFrame, global_dir: str) -> None:
    base = peak[(peak["Domain"] == "Known (Mixed Species)") & (peak["Reranking"] == "enabled") & (~peak["Optimised"])].nlargest(20, PRIMARY_METRIC).copy()
    if base.empty: return
    base["Configuration"] = base["Run_Name"].apply(get_human_readable_variant)
    out = base[["Configuration", "BaKS", "BAUS", "H-Score", PRIMARY_METRIC, SECONDARY_METRIC]].copy()
    out.columns = ["Configuration", "BaKS", "BAUS", "H-Score", "ARI", "mAP (%)"]
    _save(out, os.path.join(global_dir, "open_set_metrics_top20"))

def generate_all_global_tables(peak: pd.DataFrame, global_dir: str) -> None:
    print("  Generating global tables …")
    global_top_n(peak, global_dir)
    global_generalisation_gap(peak, global_dir)
    global_reranking_impact(peak, global_dir)
    global_hdbscan_impact(peak, global_dir)
    global_species_comparison(peak, global_dir)
    global_open_set_summary(peak, global_dir)