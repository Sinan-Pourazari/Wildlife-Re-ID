import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
import hdbscan

# Import from your existing modules
from train_test_prototype import ReIDModel, reduce_to_nd
from dataloader import UniversalGraphDataset
import embedding_clusterings as ec
import plotly.express as px
import pandas as pd

def plot_interactive_3d_tsne(xyz, labels, title, save_path):
    """Generates an interactive 3D t-SNE and saves it as an HTML file."""
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Component 1': xyz[:, 0],
        'Component 2': xyz[:, 1],
        'Component 3': xyz[:, 2],
        # Convert labels to string so Plotly treats them as discrete categories/colors
        'Identity': [str(lab) for lab in labels] 
    })

    # Create the interactive 3D scatter
    fig = px.scatter_3d(
        df, 
        x='Component 1', 
        y='Component 2', 
        z='Component 3',
        color='Identity',
        hover_name='Identity', # Tooltip when you hover over a dot
        title=f"{title}<br>Interactive 3D t-SNE Projection (N={len(labels)})",
        opacity=0.8
    )

    # 1. Shrink the dots (Plotly defaults to massive dots)
    # 2. Hide the legend (With 300+ identities, the legend will crash your browser)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=40))

    # Save as an interactive webpage
    fig.write_html(save_path)

def extract_features(model, dataloader, device):
    """Passes the test set through the model and collects embeddings and labels."""
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

def compute_reid_metrics(features, labels, device='cuda'):
    """Computes Rank-1, Rank-5, Rank-10 and mAP using cosine similarity."""
    # 1. Move to GPU for fast matrix math
    features = features.to(device)
    labels = labels.to(device)
    
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features, features.t())
    
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # 2. Bring back to CPU for the loop (so we don't clog VRAM)
    sorted_labels = sorted_labels.cpu()
    labels = labels.cpu()
    
    N = labels.size(0)
    aps = []
    cmc_1, cmc_5, cmc_10 = 0.0, 0.0, 0.0
    
    for i in range(N):
        query_label = labels[i]
        matches = (sorted_labels[i] == query_label).float()
        num_pos = (labels == query_label).sum().item() - 1 
        
        if num_pos == 0:
            continue
            
        # CMC at Rank 1, 5, 10
        if matches[0] == 1: cmc_1 += 1
        if matches[:5].sum() > 0: cmc_5 += 1
        if matches[:10].sum() > 0: cmc_10 += 1
            
        # Average Precision (AP)
        cum_matches = torch.cumsum(matches, dim=0)
        precision = cum_matches / torch.arange(1, N + 1, dtype=torch.float32)
        ap = torch.sum(precision * matches) / num_pos
        aps.append(ap.item())
        
    valid_queries = len(aps) if aps else 1
    rank_1 = (cmc_1 / valid_queries) * 100
    rank_5 = (cmc_5 / valid_queries) * 100
    rank_10 = (cmc_10 / valid_queries) * 100
    mAP = np.mean(aps) * 100 if aps else 0.0
    
    return rank_1, rank_5, rank_10, mAP

def compute_clustering_metrics(args, embeddings, labels):
    # min_cluster_size=2 is key for Re-ID where some animals have few photos
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean', core_dist_n_jobs=
                                args.workers)
    predicted_ids = clusterer.fit_predict(embeddings.numpy())
    
    ari = adjusted_rand_score(labels.numpy(), predicted_ids)
    nmi = normalized_mutual_info_score(labels.numpy(), predicted_ids)
    
    # Count clusters (ignoring -1 noise)
    discovered_ids = len(set(predicted_ids)) - (1 if -1 in predicted_ids else 0)
    return ari, nmi, discovered_ids

def plot_benchmark_results(results_df, save_dir):
    """Generates two bar charts: Retrieval (Rank/mAP) and Clustering (ARI/NMI)."""
    # 1. Retrieval Metrics Graph
    plt.figure(figsize=(14, 7))
    x = np.arange(len(results_df))
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

from matplotlib import patheffects

from matplotlib import patheffects
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_detailed_tsne(features_nd, labels, title, save_path):
    """
    Pass it a 2D array -> Saves an independent 2D PNG.
    Pass it a 3D array -> Saves an independent 3D PNG.
    """
    dim = features_nd.shape[1]
    
    fig = plt.figure(figsize=(12, 10))
    
    # Strictly creates ONE full-screen axis (No subplots)
    if dim == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()

    # Generate distinct colors
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", len(unique_labels))
    np.random.seed(42)
    np.random.shuffle(palette) 
    
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}
    color_list = [label_to_color[lab] for lab in labels]

    # Plot based on dimension
    if dim == 3:
        ax.scatter(features_nd[:, 0], features_nd[:, 1], features_nd[:, 2], c=color_list, s=15, alpha=0.8)
        ax.set_zlabel("Component 3")
    else:
        ax.scatter(features_nd[:, 0], features_nd[:, 1], c=color_list, s=15, alpha=0.8)

    ax.set_title(f"{title}\n{dim}D t-SNE Projection (N={len(labels)} | Classes={len(unique_labels)})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    # Annotate Top 15
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
    """Reproduces the dataset splitting logic from train_test_prototype.py"""
    df = pd.read_csv(args.csv_path).dropna(subset=['identity'])
    df['global_identity'] = df['dataset'] + "_" + df['identity'].astype(str)
    
    # Filter singles
    counts = df['global_identity'].value_counts()
    df = df[df['global_identity'].isin(counts[counts > 1].index)].reset_index(drop=True)
    
    from sklearn.preprocessing import LabelEncoder
    df['global_label'] = LabelEncoder().fit_transform(df['global_identity'])

    # Holdout logic
    if args.holdout_species:
        test_df = df[df['species'] == args.holdout_species].reset_index(drop=True)
    elif args.holdout_dataset:
        test_df = df[df['dataset'] == args.holdout_dataset].reset_index(drop=True)
    else:
        print("Loading exact test split from training run...")
        test_df = pd.read_csv(f"{args.checkpoints_dir}/current_test_split.csv")

    return list(zip(test_df["path"], test_df["global_label"]))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Starting Evaluation Suite...")
    
    # 1. Setup Data Paths
    eval_out_dir = os.path.join(args.checkpoints_dir, "evaluation_results")
    csv_path = os.path.join(eval_out_dir, 'benchmark_stats.csv')
    
    # --- PHASE 1: EVALUATION (OR SKIP) ---
    if args.use_existing_csv and os.path.exists(csv_path):
        print(f"\n[ SKIPPING EVALUATION: Using existing benchmark results from {csv_path} ]")
        df = pd.read_csv(csv_path)
        
        # We still need the dataloader for the t-SNE feature extraction later!
        test_samples = get_test_samples(args)

        true_unique_ids = len(set(label for _, label in test_samples))
        print(f"Evaluated Test Set Size: {len(test_samples)} images")
        print(f"--> TRUE Unique Identities in Test Set: {true_unique_ids}")
        
        test_dataset = UniversalGraphDataset(
            samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir,
            mode=args.data_mode, n_segments=args.segments, rebuild_cache=False 
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.workers, persistent_workers=True)
    else:
        test_samples = get_test_samples(args)

        true_unique_ids = len(set(label for _, label in test_samples))
        print(f"Evaluated Test Set Size: {len(test_samples)} images")
        print(f"--> TRUE Unique Identities in Test Set: {true_unique_ids}")

        test_dataset = UniversalGraphDataset(
            samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir,
            mode=args.data_mode, n_segments=args.segments, rebuild_cache=False 
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.workers, persistent_workers=True)
        
        os.makedirs(eval_out_dir, exist_ok=True)
        summary_path = os.path.join(eval_out_dir, "test_set_summary.txt")

        with open(summary_path, "w") as f:
            f.write("=== Test Set Ground Truth ===\n")
            f.write(f"Total Images: {len(test_samples)}\n")
            f.write(f"True Unique Identities: {true_unique_ids}\n")
        print(f"--> Ground truth stats saved to {summary_path}")
        pth_files = glob.glob(os.path.join(args.checkpoints_dir, "*.pth"))

        if not pth_files:
            print(f"No .pth files found in {args.checkpoints_dir}")
            return
        
        print(f"Found {len(pth_files)} checkpoints. Beginning Evaluation...")
        results = []
        
        for ckpt in sorted(pth_files):
            model_name = os.path.basename(ckpt)
            print(f"\nEvaluating: {model_name}")
            
            try:
                model, _ = ReIDModel.load(ckpt, device=device)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
                
            features, labels = extract_features(model, test_loader, device)
            
            r1, r5, r10, map_val = compute_reid_metrics(features, labels)
            ari, nmi, discovered_ids = compute_clustering_metrics(args, features, labels)
            
            print(f"  Retrieval -> R1: {r1:.1f}% | R5: {r5:.1f}% | R10: {r10:.1f}% | mAP: {map_val:.1f}%")
            print(f"  Clustering-> ARI: {ari:.3f} | NMI: {nmi:.3f} | Discovered IDs: {discovered_ids}")
            
            results.append({
                'Model Name': model_name.replace('.pth', ''),
                'Rank-1 (%)': r1, 'Rank-5 (%)': r5, 'Rank-10 (%)': r10, 'mAP (%)': map_val,
                'ARI': ari, 'NMI': nmi, 'Discovered IDs': discovered_ids,
                'ckpt_path': ckpt 
            })
            
            del model, features, labels
            torch.cuda.empty_cache()
            
        df = pd.DataFrame(results)
        
        print("\n================ BENCHMARK SUMMARY ================")
        display_df = df.drop(columns=['ckpt_path'])
        print(display_df.to_string(index=False))
        
        plot_benchmark_results(df, eval_out_dir)
        df.to_csv(csv_path, index=False)

    # --- PHASE 2: TOP-K VISUALIZATION (2D and 3D) ---
    if args.top_k_detailed > 0:
        top_k = min(args.top_k_detailed, len(df))
        print(f"\n[ Detailed Analysis: Generating t-SNE for Top {top_k} Models (Sorted by mAP) ]")
        
        top_models = df.sort_values(by='mAP (%)', ascending=False).head(top_k)
        
        for idx, row in top_models.iterrows():
            model_name = row['Model Name']
            print(f"\n--> Processing t-SNE for {model_name}...")
            
            model, _ = ReIDModel.load(row['ckpt_path'], device=device)
            features, labels = extract_features(model, test_loader, device)
            
            # --- 1. Generate & Save 2D Image ---
            print("    Computing 2D Reduction...")
            xy_2d = reduce_to_nd(args,features, n_components=2, method="tsne", seed=42)
            save_path_2d = os.path.join(eval_out_dir, f'tsne_2d_{model_name}.png')
            plot_detailed_tsne(xy_2d, labels.numpy(), model_name, save_path_2d)
            
            # --- 2. Generate 3D ---
            print("    Computing 3D Reduction...")
            xyz_3d = reduce_to_nd(args, features, n_components=3, method="tsne", seed=42)
            
            # Save the static PNG for a quick glance
            save_path_3d_png = os.path.join(eval_out_dir, f'tsne_3d_{model_name}.png')
            plot_detailed_tsne(xyz_3d, labels.numpy(), model_name, save_path_3d_png)
            
            # --- NEW: Save the Interactive HTML ---
            save_path_3d_html = os.path.join(eval_out_dir, f'tsne_3d_interactive_{model_name}.html')
            plot_interactive_3d_tsne(xyz_3d, labels.numpy(), model_name, save_path_3d_html)
            
            del model, features, labels
            torch.cuda.empty_cache()

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
    
    # GNN & Dataloader
    parser.add_argument("--segments", type=int, default=300, help="Number of superpixel segments")
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "memory", "lazy"], help="Data load mode")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--workers", type=int, default=4, help="CPU workers for DataLoader")
    
    # Evaluation specifics
    parser.add_argument("--top_k_detailed", type=int, default=5, help="Generate t-SNE embeddings and extra graphs for top K models")
    parser.add_argument("--use_existing_csv", action="store_true", help="Skip evaluation and plot t-SNE directly from benchmark_stats.csv")
    args = parser.parse_args()
    main(args)