import os
import glob
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import hdbscan
import plotly.express as px

# Import from your existing modules
from train_test_prototype import ReIDModel, reduce_to_nd
from dataloader import UniversalGraphDataset
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def calculate_baks(y_true: np.ndarray, y_pred: np.ndarray, known_classes: set) -> float:
    """
    Calculates the Balanced Accuracy on Known Samples (BaKS).
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        known_classes (set or list): The label IDs present in the training set.
        
    Returns:
        float: The BaKS score (0.0 to 1.0). Returns 0.0 if no known samples exist.
    """
    known_mask = np.isin(y_true, list(known_classes))
    y_true_known = y_true[known_mask]
    y_pred_known = y_pred[known_mask]

    # Safety check to prevent division by zero if the batch/set has no knowns
    if len(y_true_known) == 0:
        return 0.0

    return float(balanced_accuracy_score(y_true_known, y_pred_known))

import numpy as np

def calculate_baus(y_true: np.ndarray, y_pred: np.ndarray, known_classes: set, unknown_label: int = -1) -> float:
    """
    Calculates the Balanced Accuracy on Unknown Samples (BAUS).
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        known_classes (set or list): The label IDs present in the training set.
        unknown_label (int): The label your pipeline assigns to rejected predictions.
        
    Returns:
        float: The BAUS score (0.0 to 1.0). Returns 0.0 if no unknown samples exist.
    """
    unknown_mask = ~np.isin(y_true, list(known_classes))
    y_true_unknown = y_true[unknown_mask]
    y_pred_unknown = y_pred[unknown_mask]

    # Safety check if the test set is strictly closed-set (no unknowns)
    if len(y_true_unknown) == 0:
        return 0.0

    unique_unknown_classes = np.unique(y_true_unknown)
    class_rejection_rates = []

    for uk_cls in unique_unknown_classes:
        cls_mask = (y_true_unknown == uk_cls)
        
        # Calculate what percentage of this specific animal's photos were correctly rejected
        correct_rejections = (y_pred_unknown[cls_mask] == unknown_label)
        rejection_rate = np.mean(correct_rejections)
        
        class_rejection_rates.append(rejection_rate)

    # Macro-average so a rare unknown animal matters as much as a common unknown animal
    return float(np.mean(class_rejection_rates))

def compute_open_set_score(y_true, y_pred, known_classes):
    baks = calculate_baks(y_true, y_pred, known_classes)
    baus = calculate_baus(y_true, y_pred, known_classes)
    
    geometric_mean = np.sqrt(baks * baus)
    
    return baks, baus, geometric_mean
def plot_interactive_3d_tsne(xyz, labels, title, save_path):
    df = pd.DataFrame({
        'Component 1': xyz[:, 0],
        'Component 2': xyz[:, 1],
        'Component 3': xyz[:, 2],
        'Identity': [str(lab) for lab in labels] 
    })

    fig = px.scatter_3d(
        df, x='Component 1', y='Component 2', z='Component 3',
        color='Identity', hover_name='Identity',
        title=f"{title}<br>Interactive 3D t-SNE Projection (N={len(labels)})",
        opacity=0.8
    )

    fig.update_traces(marker=dict(size=3))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(save_path)

def extract_features(model, dataloader, device):
    model.eval()
    all_emb = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting Features", leave=False):
            data = data.to(device)
            emb = model(data)
            all_emb.append(emb.cpu())
            all_labels.append(data.y.cpu())
            
    return torch.cat(all_emb), torch.cat(all_labels)

def compute_reid_metrics(features, labels, device='cpu', sim_thresh=0.6):
    features = features.to(device)
    labels = labels.to(device)
    
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features, features.t())
    
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]

    # 1. Get the absolute highest similarity score for each query
    top1_sims = sim_matrix.max(dim=1).values
    top1_labels = sorted_labels[:, 0]
    
    # 2. Apply the threshold: if the closest match is too far, label it -1 (Unknown)
    y_pred = top1_labels.clone()
    y_pred[top1_sims < sim_thresh] = -1 
    
    # 3. Move to CPU/Numpy for the scikit-learn math
    y_true_np = labels.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    N = labels.size(0)
    aps = []
    cmc_1, cmc_5, cmc_10 = 0.0, 0.0, 0.0
    
    for i in range(N):
        query_label = labels[i]
        matches = (sorted_labels[i] == query_label).float()
        num_pos = (labels == query_label).sum().item() - 1 
        
        if num_pos == 0:
            continue
            
        if matches[0] == 1: cmc_1 += 1
        if matches[:5].sum() > 0: cmc_5 += 1
        if matches[:10].sum() > 0: cmc_10 += 1
            
        cum_matches = torch.cumsum(matches, dim=0)
        precision = cum_matches / torch.arange(1, N + 1, dtype=torch.float32, device=device)
        ap = torch.sum(precision * matches) / num_pos
        aps.append(ap.item())
        
    valid_queries = len(aps) if aps else 1
    rank_1 = (cmc_1 / valid_queries) * 100
    rank_5 = (cmc_5 / valid_queries) * 100
    rank_10 = (cmc_10 / valid_queries) * 100
    mAP = np.mean(aps) * 100 if aps else 0.0
    
    # Calculate the Open-Set Metrics
    baks = calculate_baks(y_true_np, y_pred_np, known_classes)
    baus = calculate_baus(y_true_np, y_pred_np, known_classes)
    return rank_1, rank_5, rank_10, mAP, baks, baus

def compute_clustering_metrics(args, embeddings, labels):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean', core_dist_n_jobs=1)
    predicted_ids = clusterer.fit_predict(embeddings)
    
    ari = adjusted_rand_score(labels, predicted_ids)
    nmi = normalized_mutual_info_score(labels, predicted_ids)
    
    discovered_ids = len(set(predicted_ids)) - (1 if -1 in predicted_ids else 0)
    return ari, nmi, discovered_ids

def plot_benchmark_results(results_df, save_dir):
    x = np.arange(len(results_df))
    
    # 1. Retrieval Metrics Graph
    plt.figure(figsize=(14, 7))
    width = 0.2
    plt.bar(x - width*1.5, results_df['Rank-1 (%)'], width, label='Rank-1', color='#1f77b4')
    plt.bar(x - width*0.5, results_df['Rank-5 (%)'], width, label='Rank-5', color='#2ca02c')
    plt.bar(x + width*0.5, results_df['Rank-10 (%)'], width, label='Rank-10', color='#9467bd')
    plt.bar(x + width*1.5, results_df['mAP (%)'], width, label='mAP', color='#ff7f0e')
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Re-ID Model Benchmarks (Retrieval)', fontsize=14)
    plt.xticks(x, results_df['Model Name'], rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_retrieval.png'), dpi=300)
    plt.close()

    # 2. Clustering Metrics Graph
    plt.figure(figsize=(14, 7))
    width = 0.35
    plt.bar(x - width/2, results_df['ARI'], width, label='ARI', color='#8c564b')
    plt.bar(x + width/2, results_df['NMI'], width, label='NMI', color='#e377c2')
    plt.ylabel('Score (0 to 1)', fontsize=12)
    plt.title('Re-ID Model Benchmarks (Open-Set Clustering)', fontsize=14)
    plt.xticks(x, results_df['Model Name'], rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_clustering.png'), dpi=300)
    plt.close()

    if 'BaKS' in results_df.columns:
        plt.figure(figsize=(14, 7))
        width = 0.25
        plt.bar(x - width, results_df['BaKS'], width, label='BaKS (Knowns)', color='#17becf')
        plt.bar(x, results_df['BAUS'], width, label='BAUS (Unknowns)', color='#bcbd22')
        plt.bar(x + width, results_df['H-Score'], width, label='H-Score (Geometric Mean)', color='#7f7f7f')
        plt.ylabel('Score (0 to 1)', fontsize=12)
        plt.title('Re-ID Model Benchmarks (Open-Set Performance)', fontsize=14)
        plt.xticks(x, results_df['Model Name'], rotation=45, ha='right', fontsize=10)
        
        # Add a threshold line to easily spot perfect balance
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'benchmark_openset.png'), dpi=300)
        plt.close()

def plot_detailed_tsne(features_nd, labels, title, save_path):
    dim = features_nd.shape[1]
    fig = plt.figure(figsize=(12, 10))
    
    if dim == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()

    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))
    np.random.seed(42)
    np.random.shuffle(palette) 
    
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}
    color_list = [label_to_color[lab] for lab in labels]

    if dim == 3:
        ax.scatter(features_nd[:, 0], features_nd[:, 1], features_nd[:, 2], c=color_list, s=15, alpha=0.8)
        ax.set_zlabel("Component 3")
    else:
        ax.scatter(features_nd[:, 0], features_nd[:, 1], c=color_list, s=15, alpha=0.8)

    ax.set_title(f"{title}\n{dim}D t-SNE Projection (N={len(labels)} | Classes={len(unique_labels)})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    vals, counts = np.unique(labels, return_counts=True)
    top = vals[np.argsort(-counts)][:15]
    
    for lab in top:
        mask = labels == lab
        centroid = features_nd[mask].mean(axis=0)
        
        if dim == 3:
            txt = ax.text(centroid[0], centroid[1], centroid[2], str(lab), 
                          fontsize=10, weight='bold', color=label_to_color[lab])
        else:
            txt = ax.text(centroid[0], centroid[1], str(lab), 
                          fontsize=10, weight='bold', color=label_to_color[lab])
            
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_test_samples(args):
    df = pd.read_csv(args.csv_path).dropna(subset=['identity'])
    df['global_identity'] = df['dataset'] + "_" + df['identity'].astype(str)
    
    counts = df['global_identity'].value_counts()
    df = df[df['global_identity'].isin(counts[counts > 1].index)].reset_index(drop=True)
    
    from sklearn.preprocessing import LabelEncoder
    df['global_label'] = LabelEncoder().fit_transform(df['global_identity'])

    if args.holdout_species:
        test_df = df[df['species'] == args.holdout_species].reset_index(drop=True)
    elif args.holdout_dataset:
        test_df = df[df['dataset'] == args.holdout_dataset].reset_index(drop=True)
    else:
        test_df = pd.read_csv(f"{args.checkpoints_dir}/current_test_split.csv")

    if args.eval_filter_species:
        test_df = test_df[test_df['species'] == args.eval_filter_species].reset_index(drop=True)
        if len(test_df) == 0:
            raise ValueError(f"No images found for species '{args.eval_filter_species}'!")

    return list(zip(test_df["path"], test_df["global_label"]))

# ==========================================
# PARALLEL WORKER FUNCTIONS (PURE CPU)
# ==========================================

def evaluate_metrics_worker(ckpt_path, feats_np, labels_np,known_classes, args):
    """Worker function to compute metrics purely from numpy arrays on CPU"""
    model_name = os.path.basename(ckpt_path)
    
    # Reconstruct isolated CPU PyTorch tensors for the matrix math
    features = torch.from_numpy(feats_np)
    labels = torch.from_numpy(labels_np)
    
    # Pass known_classes down!
    r1, r5, r10, map_val, baks, baus = compute_reid_metrics(features, labels, known_classes, device='cpu', sim_thresh=0.6)
    ari, nmi, discovered_ids = compute_clustering_metrics(args, feats_np, labels_np)
    
    harmonic_score = np.sqrt(baks * baus)

    return {
        'Model Name': model_name.replace('.pth', ''),
        'Rank-1 (%)': r1, 'Rank-5 (%)': r5, 'Rank-10 (%)': r10, 'mAP (%)': map_val,
        'BaKS': baks, 'BAUS': baus, 'H-Score': harmonic_score, # <--- NEW METRICS ADDED HERE
        'ARI': ari, 'NMI': nmi, 'Discovered IDs': discovered_ids,
        'ckpt_path': ckpt_path 
    }

def generate_tsne_worker(ckpt_path, feats_np, labels_np, args, eval_out_dir):
    """Worker function to compute and plot t-SNE from numpy arrays on CPU"""
    model_name = os.path.basename(ckpt_path).replace('.pth', '')
    
    # reduce_to_nd usually expects a torch tensor, adapt based on your train_test_prototype
    features = torch.from_numpy(feats_np)
    
    # 2D t-SNE
    xy_2d = reduce_to_nd(args, features, n_components=2, method="tsne", seed=42)
    save_path_2d = os.path.join(eval_out_dir, f'tsne_2d_{model_name}.png')
    plot_detailed_tsne(xy_2d, labels_np, model_name, save_path_2d)
    
    # 3D t-SNE
    xyz_3d = reduce_to_nd(args, features, n_components=3, method="tsne", seed=42)
    save_path_3d_png = os.path.join(eval_out_dir, f'tsne_3d_{model_name}.png')
    plot_detailed_tsne(xyz_3d, labels_np, model_name, save_path_3d_png)
    
    save_path_3d_html = os.path.join(eval_out_dir, f'tsne_3d_interactive_{model_name}.html')
    plot_interactive_3d_tsne(xyz_3d, labels_np, model_name, save_path_3d_html)
    
    return model_name

# ==========================================
# MAIN EXECUTION
# ==========================================

def main(args):
    main_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{main_device.type.upper()}] Starting Asynchronous Pipeline...")
    
    eval_out_dir = os.path.join(args.checkpoints_dir, "evaluation_results")
    csv_path = os.path.join(eval_out_dir, 'benchmark_stats.csv')
    os.makedirs(eval_out_dir, exist_ok=True)
    
    test_samples = get_test_samples(args)
    true_unique_ids = len(set(label for _, label in test_samples))
    print(f"Evaluated Test Set Size: {len(test_samples)} images")
    print(f"--> TRUE Unique Identities in Test Set: {true_unique_ids}")
    
    pth_files = glob.glob(os.path.join(args.checkpoints_dir, "*.pth"))
    if not pth_files:
        print(f"No .pth files found in {args.checkpoints_dir}")
        return

    # Dictionary to hold pure raw NumPy arrays in RAM (Safe for Multiprocessing)
    extracted_data = {}

    # --- PHASE 1: SEQUENTIAL GPU INFERENCE ---
    if args.use_existing_csv and os.path.exists(csv_path):
        print(f"\n[ PHASE 1 SKIPPED: Using existing benchmark results ]")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n--- PHASE 1: SEQUENTIAL GPU FEATURE EXTRACTION ---")
        test_dataset = UniversalGraphDataset(
            samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir,
            mode=args.data_mode, n_segments=args.segments, rebuild_cache=False, features=args.features 
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.workers, persistent_workers=True)

# --- PHASE 1: SEQUENTIAL GPU FEATURE EXTRACTION ---
        # ... (loader setup code) ...
        for ckpt in sorted(pth_files):
            model_name = os.path.basename(ckpt)
            print(f"Processing: {model_name}")
            try:
                # Capture the second return value (train_classes)
                model, train_classes = ReIDModel.load(ckpt, device=main_device)
                features, labels = extract_features(model, test_loader, main_device)
                
                # CRITICAL: Store the known classes as a set alongside the numpy arrays
                extracted_data[ckpt] = (features.cpu().numpy(), labels.cpu().numpy(), set(train_classes))
                
                # Free VRAM immediately
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to infer {model_name}: {e}")

        # --- PHASE 2: PARALLEL CPU METRICS ---
        print(f"\n--- PHASE 2: PARALLEL METRICS CALCULATION ({args.parallel_workers} Workers) ---")
        results = []
        
        spawn_context = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=spawn_context) as executor:
            futures = {}
            # Unpack the known_classes here
            for ckpt, (feats_np, lbls_np, known_classes) in extracted_data.items():
                
                # Send raw numpy arrays AND known_classes to the worker
                future = executor.submit(evaluate_metrics_worker, ckpt, feats_np, lbls_np, known_classes, args)
                futures[future] = ckpt
                
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Metrics"):
                results.append(future.result())
                    
        df = pd.DataFrame(results)
        print("\n================ BENCHMARK SUMMARY ================")
        display_df = df.drop(columns=['ckpt_path'])
        print(display_df.to_string(index=False))
        
        plot_benchmark_results(df, eval_out_dir)
        df.to_csv(csv_path, index=False)

    # --- PHASE 3: PARALLEL TOP-K VISUALIZATION ---
    if args.top_k_detailed > 0:
        top_k = min(args.top_k_detailed, len(df))
        print(f"\n--- PHASE 3: PARALLEL t-SNE GENERATION (Top {top_k} Models) ---")
        top_models = df.sort_values(by='H-Score', ascending=False).head(top_k)
        
        # Catch for use_existing_csv scenario
        if args.use_existing_csv and not extracted_data:
            print("Re-extracting features sequentially for Top-K models...")
            test_dataset = UniversalGraphDataset(
                samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir,
                mode=args.data_mode, n_segments=args.segments, rebuild_cache=False, features=args.features 
            )
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            for row in top_models.itertuples():
                # --- FIX: Capture train_classes here ---
                model, train_classes = ReIDModel.load(row.ckpt_path, device=main_device)
                feats, lbls = extract_features(model, test_loader, main_device)
                
                # --- FIX: Store train_classes in the tuple ---
                extracted_data[row.ckpt_path] = (feats.cpu().numpy(), lbls.cpu().numpy(), set(train_classes))
                
                del model
                torch.cuda.empty_cache()

        spawn_context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=spawn_context) as executor:
            tsne_futures = []
            for row in top_models.itertuples():
                feats_np, lbls_np = extracted_data[row.ckpt_path]
                tsne_futures.append(
                    executor.submit(generate_tsne_worker, row.ckpt_path, feats_np, lbls_np, args, eval_out_dir)
                )
                
            for future in tqdm(as_completed(tsne_futures), total=len(tsne_futures), desc="Generating Plots"):
                future.result()

    print(f"\n--> All evaluations complete. Outputs saved to: {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Re-ID Comprehensive Evaluator")
    
    # Directory & Data Config
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_long_run_v2", help="Directory containing .pth models")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv", help="Dataset metadata CSV")
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k", help="Base directory for image files")
    parser.add_argument("--cache_dir", type=str, default="src/images/reid-10k/graph_cache_pool", help="Cache directory for PT graphs")
    
    # Holdout Config (Mirrors train_test_prototype)
    parser.add_argument("--holdout_dataset", type=str, default=None, help="Evaluate specifically on held-out dataset")
    parser.add_argument("--holdout_species", type=str, default=None, help="Evaluate specifically on held-out species")
    parser.add_argument("--eval_filter_species", type=str, default=None, help="Isolate a single species from the standard test split (e.g., 'tiger')")
    
    # GNN & Dataloader
    parser.add_argument("--segments", type=int, default=300, help="Number of superpixel segments")
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "memory", "lazy"], help="Data load mode")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--workers", type=int, default=4, help="CPU workers for DataLoader")
    
    # Evaluation specifics
    parser.add_argument("--top_k_detailed", type=int, default=5, help="Generate t-SNE embeddings and extra graphs for top K models")
    parser.add_argument("--use_existing_csv", action="store_true", help="Skip evaluation and plot t-SNE directly from benchmark_stats.csv")
    parser.add_argument("--parallel_workers", type=int, default=4, help="Number of CPU/Evaluation processes to run concurrently")
    
    parser.add_argument("--features", nargs="+", default=["color", "pos", "hog", "lbp", "texture"], help="List of node features to extract")

    args = parser.parse_args()
    main(args)