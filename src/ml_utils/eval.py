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

# Import from your existing modules
from train_test_prototype import ReIDModel, reduce_to_2d
from dataloader import UniversalGraphDataset
import embedding_clusterings as ec

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

def compute_reid_metrics(features, labels):
    """Computes Rank-1, Rank-5, Rank-10 and mAP using cosine similarity."""
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features, features.t())
    
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]
    
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

def compute_clustering_metrics(embeddings, labels, threshold=0.30):
    """Computes Open-Set metrics: ARI, NMI using IdentityMemory."""
    memory = ec.IdentityMemory(threshold=threshold, max_exemplars_per_identity=5)
    
    predicted_ids = []
    for emb in embeddings:
        pid, _, _ = memory.upsert(emb)
        predicted_ids.append(pid)
        
    ari = adjusted_rand_score(labels.numpy(), predicted_ids)
    nmi = normalized_mutual_info_score(labels.numpy(), predicted_ids)
    
    return ari, nmi, len(memory.memory)

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

def plot_detailed_tsne(xy, labels, title, save_path):
    """Generates a t-SNE plot for top-performing models."""
    plt.figure(figsize=(12, 10))
    unique = np.unique(labels)
    label_to_idx = {lab: i for i, lab in enumerate(unique)}
    c = np.array([label_to_idx[lab] for lab in labels], dtype=int)

    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=c, cmap='tab20', s=15, alpha=0.8)
    plt.title(f"{title}\nt-SNE Projection (N={len(labels)} | Classes={len(unique)})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # Annotate top 15 clusters for clarity
    vals, counts = np.unique(labels, return_counts=True)
    top = vals[np.argsort(-counts)][:15]
    for lab in top:
        mask = labels == lab
        cx, cy = xy[mask].mean(axis=0)
        plt.text(cx, cy, str(lab), fontsize=10, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

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
        unique_labels = df["global_label"].unique()
        _, test_labels = train_test_split(unique_labels, test_size=0.2, random_state=42)
        test_df = df[df["global_label"].isin(test_labels)].reset_index(drop=True)

    return list(zip(test_df["path"], test_df["global_label"]))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device.type.upper()}] Starting Evaluation Suite...")
    
    test_samples = get_test_samples(args)
    print(f"Evaluated Test Set Size: {len(test_samples)} images")

    test_dataset = UniversalGraphDataset(
        samples=test_samples,
        root_dir=args.root_dir,
        cache_dir=args.cache_dir,
        mode=args.data_mode,
        n_segments=args.segments,
        rebuild_cache=False 
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)
    
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
        ari, nmi, discovered_ids = compute_clustering_metrics(features, labels)
        
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
    
    # Save base charts and CSV
    eval_out_dir = os.path.join(args.checkpoints_dir, "evaluation_results")
    os.makedirs(eval_out_dir, exist_ok=True)
    
    plot_benchmark_results(df, eval_out_dir)
    df.to_csv(os.path.join(eval_out_dir, 'benchmark_stats.csv'), index=False)
    
    # ==========================================
    # Top-K Detailed Analysis (t-SNE Embeddings)
    # ==========================================
    if args.top_k_detailed > 0:
        top_k = min(args.top_k_detailed, len(df))
        print(f"\n[ Detailed Analysis: Generating t-SNE for Top {top_k} Models (Sorted by mAP) ]")
        
        top_models = df.sort_values(by='mAP (%)', ascending=False).head(top_k)
        
        for idx, row in top_models.iterrows():
            model_name = row['Model Name']
            print(f"--> Processing t-SNE for {model_name}...")
            
            model, _ = ReIDModel.load(row['ckpt_path'], device=device)
            features, labels = extract_features(model, test_loader, device)
            
            # Reduce and Plot
            xy_2d = reduce_to_2d(features, method="tsne", seed=42)
            save_path = os.path.join(eval_out_dir, f'tsne_{model_name}.png')
            
            plot_detailed_tsne(xy_2d, labels.numpy(), model_name, save_path)
            
            del model, features, labels
            torch.cuda.empty_cache()

    print(f"\n--> All evaluations complete. Outputs saved to: {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Re-ID Comprehensive Evaluator")
    
    # Directory & Data Config
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_long_run", help="Directory containing .pth models")
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
    
    args = parser.parse_args()
    main(args)