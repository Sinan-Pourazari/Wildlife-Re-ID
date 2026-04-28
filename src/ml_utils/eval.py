import os
import glob
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
from tqdm import tqdm
import plotly.express as px

# Scikit-Learn & Scipy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Graph Clustering
import hdbscan
import igraph as ig
import leidenalg as la

# Custom Modules
from train_test_prototype import ReIDModel, reduce_to_nd
from dataloader import UniversalGraphDataset
from torch_geometric.loader import DataLoader
import re
# =====================================================================
# 1. FEATURE EXTRACTION & DATA UTILS
# =====================================================================
def get_test_samples(args):
    test_df = pd.read_csv(args.csv_path, low_memory=False)
    print(f"--> Initial CSV loaded. Total rows: {len(test_df)}")

    # 1. Filter by Dataset Name
    if args.holdout_dataset:
        if 'dataset' in test_df.columns:
            # Strip whitespace and make lowercase for safe comparison
            safe_target = args.holdout_dataset.strip().lower()
            test_df['dataset_safe'] = test_df['dataset'].astype(str).str.strip().str.lower()
            
            test_df = test_df[test_df['dataset_safe'] == safe_target].reset_index(drop=True)
            print(f"--> After --holdout_dataset '{args.holdout_dataset}': {len(test_df)} rows remain.")
            
            if len(test_df) == 0:
                # Print the actual datasets available to help debug!
                unique_ds = pd.read_csv(args.csv_path, low_memory=False)['dataset'].dropna().unique()
                raise ValueError(f"No images found for dataset '{args.holdout_dataset}'. Available datasets in CSV: {unique_ds}")
        else:
            print(f"--> WARNING: 'dataset' column not found.")

    # 2. Filter by Species
    if args.holdout_species:
        if 'species' in test_df.columns:
            safe_target = args.holdout_species.strip().lower()
            test_df['species_safe'] = test_df['species'].astype(str).str.strip().str.lower()
            test_df = test_df[test_df['species_safe'] == safe_target].reset_index(drop=True)
            print(f"--> After --holdout_species '{args.holdout_species}': {len(test_df)} rows remain.")
        else:
            print(f"--> WARNING: 'species' column not found.")

    # =================================================================
    # 3 & 4. IDENTITY FILTERING (Bypass for Submission!)
    # =================================================================
    if getattr(args, 'generate_submission', False):
        # SUBMISSION MODE: Keep all images, mock the identities
        if 'identity' not in test_df.columns:
            test_df['identity'] = "unknown"
        test_df['identity'] = test_df['identity'].fillna("unknown")
        print(f"--> Submission Mode: Kept all {len(test_df)} rows (bypassing identity checks).")
    else:
        # VALIDATION MODE: Require strict ground truth identities
        if 'identity' not in test_df.columns and 'animal_id' in test_df.columns:
            test_df['identity'] = test_df['animal_id'].astype(str)
            
        #test_df = test_df[test_df['identity'] != 'unknown'].dropna(subset=['identity']).reset_index(drop=True)
        #print(f"--> After dropping 'unknown' or missing identities: {len(test_df)} rows remain.")

        if len(test_df) == 0:
            raise ValueError("All images were dropped because their identity was 'unknown' or missing.")

        # 4. Filter out singletons (Re-ID metrics require at least 2 images per ID)
        #counts = test_df['identity'].value_counts()
        #keep_ids = counts[counts > 1].index
        #test_df = test_df[test_df['identity'].isin(keep_ids)].reset_index(drop=True)
        print(f"--> After dropping singleton identities (IDs with only 1 image): {len(test_df)} rows remain.")

        if len(test_df) == 0:
            raise ValueError("All images were dropped because every identity only had 1 image (singletons). Re-ID evaluation requires >= 2 images per identity!")

    # 5. Generate integer labels required by PyG DataLoader
    from sklearn.preprocessing import LabelEncoder
    test_df['global_label'] = LabelEncoder().fit_transform(test_df['identity'])
        
    if 'species' in test_df.columns:
        test_df['species_label'] = LabelEncoder().fit_transform(test_df['species'].astype(str))
    else:
        test_df['species_label'] = 0

    # 6. Optional: Cap Max Images
    if hasattr(args, 'max_images_per_id') and args.max_images_per_id is not None:
        test_df = test_df.groupby('global_label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.max_images_per_id), random_state=42)
        ).reset_index(drop=True)
        print(f"--> Downsampled test set to max {args.max_images_per_id} images per identity. Final size: {len(test_df)}")

    return test_df

def extract_features(model, dataloader, device, species_confidence_thresh=0.8):
    model.eval()
    all_emb, all_labels, all_species_preds, all_species_labels = [], [], [], []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting Features", leave=False):
            data = data.to(device)
            labels = data.y.view(-1, 2)[:, 0] if data.y.dim() > 1 else data.y.view(-1)
            
            with torch.amp.autocast(device_type="cuda"):
                out = model(data)
            
            if isinstance(out, tuple):
                emb, species_logits = out
                probs = F.softmax(species_logits, dim=1)
                max_probs, species_preds = torch.max(probs, dim=1)
                all_species_preds.append(species_preds.cpu())
            else:
                emb = out
                all_species_preds.append(torch.zeros(emb.size(0), dtype=torch.long))
                
            labels = data.y.view(emb.size(0), -1)[:, 0]
            species_labels = data.y.view(emb.size(0), -1)[:, 1] 
            
            all_emb.append(emb.cpu())
            all_species_labels.append(species_labels.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_emb), torch.cat(all_labels), torch.cat(all_species_preds), torch.cat(all_species_labels)            

# =====================================================================
# 2. METRICS (Re-ID & Open-Set)
# =====================================================================
def calculate_baks(y_true: np.ndarray, y_pred: np.ndarray, known_classes: set) -> float:
    known_mask = np.isin(y_true, list(known_classes))
    if len(y_true[known_mask]) == 0: return 0.0
    return float(balanced_accuracy_score(y_true[known_mask], y_pred[known_mask]))

def calculate_baus(y_true: np.ndarray, y_pred: np.ndarray, known_classes: set, unknown_label: int = -1) -> float:
    unknown_mask = ~np.isin(y_true, list(known_classes))
    y_true_unknown = y_true[unknown_mask]
    if len(y_true_unknown) == 0: return 0.0

    class_rejection_rates = []
    for uk_cls in np.unique(y_true_unknown):
        cls_mask = (y_true_unknown == uk_cls)
        correct_rejections = (y_pred[unknown_mask][cls_mask] == unknown_label)
        class_rejection_rates.append(np.mean(correct_rejections))

    return float(np.mean(class_rejection_rates))

def compute_reid_metrics(features, labels, known_classes, device='cpu', sim_thresh=1.5):
    features = features.to(device)
    labels = labels.to(device)
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features, features.t())
    
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]

    top1_sims = sim_matrix.max(dim=1).values
    top1_labels = sorted_labels[:, 0]
    
    y_pred = top1_labels.clone()
    y_pred[top1_sims < sim_thresh] = -1 
    
    y_true_np, y_pred_np = labels.cpu().numpy(), y_pred.cpu().numpy()
    N = labels.size(0)
    aps = []
    cmc_1, cmc_5, cmc_10 = 0.0, 0.0, 0.0
    
    for i in range(N):
        query_label = labels[i]
        matches = (sorted_labels[i] == query_label).float()
        num_pos = (labels == query_label).sum().item() - 1 
        
        if num_pos == 0: continue
            
        if matches[0] == 1: cmc_1 += 1
        if matches[:5].sum() > 0: cmc_5 += 1
        if matches[:10].sum() > 0: cmc_10 += 1
            
        cum_matches = torch.cumsum(matches, dim=0)
        precision = cum_matches / torch.arange(1, N + 1, dtype=torch.float32, device=device)
        aps.append((torch.sum(precision * matches) / num_pos).item())
        
    valid_queries = len(aps) if aps else 1
    mAP = np.mean(aps) * 100 if aps else 0.0
    
    baks = calculate_baks(y_true_np, y_pred_np, known_classes)
    baus = calculate_baus(y_true_np, y_pred_np, known_classes)
    return (cmc_1/valid_queries)*100, (cmc_5/valid_queries)*100, (cmc_10/valid_queries)*100, mAP, baks, baus

# =====================================================================
# 3. GRAPH CLUSTERING (Leiden & k-Reciprocal)
# =====================================================================
def k_reciprocal_rerank(sim_matrix, k1=20, k2=6, lambda_value=0.3):
    device = sim_matrix.device
    N = sim_matrix.size(0)
    
    _, topk_indices = torch.topk(sim_matrix, k=k1, dim=1)
    knn_matrix = torch.zeros((N, N), dtype=torch.bool, device=device)
    knn_matrix.scatter_(1, topk_indices, True)
    
    mutual_matrix = knn_matrix & knn_matrix.t()
    mutual_float = mutual_matrix.float()
    
    intersection = torch.mm(mutual_float, mutual_float.t())
    sizes = mutual_float.sum(dim=1)
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - intersection
    union[union == 0] = 1e-9
    
    jaccard_sim = intersection / union
    final_sim = (1 - lambda_value) * sim_matrix + lambda_value * jaccard_sim
    return final_sim

def predict_clusters_leiden(embeddings, species_preds, sim_thresh=0.50, k1=10, lambda_val=0.3):
    features = torch.tensor(embeddings, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features, features.t())
    
    sim_matrix = k_reciprocal_rerank(sim_matrix, k1=k1, lambda_value=lambda_val)
    
    #species_preds_t = torch.tensor(species_preds)
    #cross_species_mask = species_preds_t.unsqueeze(1) != species_preds_t.unsqueeze(0)
    #sim_matrix[cross_species_mask] = -1.0 
    
    sim_matrix_np = sim_matrix.numpy()
    sources, targets = np.where(sim_matrix_np > sim_thresh)
    weights = sim_matrix_np[sources, targets]
    
    g = ig.Graph(n=len(embeddings), edges=list(zip(sources, targets)), directed=False)
    g.es['weight'] = weights
    partition = la.find_partition(g, la.ModularityVertexPartition, weights=g.es['weight'])
    
    return np.array(partition.membership)

def compute_clustering_metrics_leiden(args, embeddings, labels, species_preds, sim_thresh=0.50, k1=10, lambda_val=0.3):
    predicted_ids = predict_clusters_leiden(embeddings, species_preds, sim_thresh, k1, lambda_val)
    n_components = len(set(predicted_ids))
    ari = adjusted_rand_score(labels, predicted_ids)
    nmi = normalized_mutual_info_score(labels, predicted_ids)
    return ari, nmi, n_components

# =====================================================================
# 4. EXHAUSTIVE TUNING
# =====================================================================
def tune_leiden_hyperparameters(embeddings, labels, species_preds, disable_pbar=False):
    """Grid Search over sim_thresh, k1, and lambda."""
    if not disable_pbar: print("\n[ Tuning ] Running Leiden Hyperparameter Grid Search...")
    true_ids_count = len(np.unique(labels))

    best_ari_score = -1
    best_params_ari = {}

    best_id_diff = float('inf')
    best_id_ari = -1
    best_params_ids = {}

    # FINER GRAINED SEARCH SPACE
    thresholds = np.arange(0.30, 0.92, 0.02)
    k1_values = [10, 15, 20]
    lambda_values = [0.2, 0.3, 0.4]

    features = torch.tensor(embeddings, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)
    base_sim_matrix = torch.mm(features, features.t())

    species_preds_t = torch.tensor(species_preds)
    cross_species_mask = species_preds_t.unsqueeze(1) != species_preds_t.unsqueeze(0)

    total_combinations = len(k1_values) * len(lambda_values) * len(thresholds)
    pbar = tqdm(total=total_combinations, desc="Testing Combinations", leave=False, disable=disable_pbar)

    for k1 in k1_values:
        for lamb in lambda_values:
            reranked_sim = k_reciprocal_rerank(base_sim_matrix.clone(), k1=k1, lambda_value=lamb)
            #reranked_sim[cross_species_mask] = -1.0 
            reranked_sim_np = reranked_sim.numpy()

            for thresh in thresholds:
                sources, targets = np.where(reranked_sim_np > thresh)
                weights = reranked_sim_np[sources, targets]
                
                if len(sources) > 0:
                    g = ig.Graph(n=len(embeddings), edges=list(zip(sources, targets)), directed=False)
                    g.es['weight'] = weights
                    partition = la.find_partition(g, la.ModularityVertexPartition, weights=g.es['weight'])
                    predicted_ids = np.array(partition.membership)
                else:
                    predicted_ids = np.arange(len(embeddings))
                
                ari = adjusted_rand_score(labels, predicted_ids)
                nmi = normalized_mutual_info_score(labels, predicted_ids)
                discovered_ids = len(set(predicted_ids))
                id_diff = abs(discovered_ids - true_ids_count)
                
                current_params = {'thresh': thresh, 'k1': k1, 'lambda': lamb, 'ari': ari, 'nmi': nmi, 'discovered_ids': discovered_ids}

                if ari > best_ari_score:
                    best_ari_score = ari
                    best_params_ari = current_params.copy()
                
                if id_diff < best_id_diff or (id_diff == best_id_diff and ari > best_id_ari):
                    best_id_diff = id_diff
                    best_id_ari = ari
                    best_params_ids = current_params.copy()

                pbar.update(1)
    pbar.close()
    
    if not disable_pbar:
        print("\n" + "="*50)
        print("🏆 BEST PARAMS: PURE ARI OPTIMIZATION 🏆")
        print(f"sim_thresh : {best_params_ari['thresh']:.2f} | k1 : {best_params_ari['k1']} | lambda : {best_params_ari['lambda']:.2f}")
        print(f"--> ARI    : {best_params_ari['ari']:.4f} | IDs: {best_params_ari['discovered_ids']} (Actual: {true_ids_count})")
        print("-" * 50)
        print("🎯 BEST PARAMS: CLOSEST IDENTITY COUNT 🎯")
        print(f"sim_thresh : {best_params_ids['thresh']:.2f} | k1 : {best_params_ids['k1']} | lambda : {best_params_ids['lambda']:.2f}")
        print(f"--> ARI    : {best_params_ids['ari']:.4f} | IDs: {best_params_ids['discovered_ids']} (Actual: {true_ids_count})")
        print("="*50 + "\n")
    
    return best_params_ari, best_params_ids

# =====================================================================
# 5. VISUALIZATION & PLOTTING
# =====================================================================
def plot_benchmark_results(results_df, save_dir):
    """Plots the Baseline results from Phase 2"""
    x = np.arange(len(results_df))
    
    # 1. Retrieval
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

    # 2. Baseline Clustering
    plt.figure(figsize=(14, 7))
    width = 0.35
    plt.bar(x - width/2, results_df['Baseline ARI'], width, label='Baseline ARI', color='#8c564b')
    plt.bar(x + width/2, results_df['Baseline NMI'], width, label='Baseline NMI', color='#e377c2')
    plt.ylabel('Score (0 to 1)', fontsize=12)
    plt.title('Re-ID Model Benchmarks (Baseline Clustering)', fontsize=14)
    plt.xticks(x, results_df['Model Name'], rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_clustering.png'), dpi=300)
    plt.close()

def plot_tuned_clustering(tuning_df, save_dir):
    """Plots the deeply tuned comparisons from Phase 2.5"""
    x = np.arange(len(tuning_df))
    plt.figure(figsize=(12, 7))
    width = 0.35
    plt.bar(x - width/2, tuning_df['ARI (Pure)'], width, label='ARI (Optimized for Pure Cluster Quality)', color='#8c564b')
    plt.bar(x + width/2, tuning_df['ARI (ID-Match)'], width, label='ARI (Optimized for True Identity Count)', color='#17becf')
    plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
    plt.title('Top Models: Leiden Hyperparameter Tuning Comparison', fontsize=14)
    plt.xticks(x, tuning_df['Model Name'], rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tuning_clustering_comparison.png'), dpi=300)
    plt.close()

def plot_detailed_tsne(features_nd, labels, title, save_path):
    dim = features_nd.shape[1]
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d') if dim == 3 else plt.axes()

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
    for lab in vals[np.argsort(-counts)][:15]:
        mask = labels == lab
        centroid = features_nd[mask].mean(axis=0)
        txt = ax.text(*centroid, str(lab), fontsize=10, weight='bold', color=label_to_color[lab])
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_interactive_3d_tsne(xyz, labels, title, save_path):
    df = pd.DataFrame({'Component 1': xyz[:, 0], 'Component 2': xyz[:, 1], 'Component 3': xyz[:, 2], 'Identity': [str(lab) for lab in labels]})
    fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3', color='Identity', hover_name='Identity', title=f"{title}<br>Interactive 3D t-SNE Projection", opacity=0.8)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=40))
    fig.write_html(save_path)

# =====================================================================
# 6. PARALLEL WORKERS
# =====================================================================
def evaluate_metrics_worker(ckpt_path, feats_np, labels_np, species_preds_np, known_classes, args, species_labels_np):
    """PHASE 2: Fast Static Baseline Evaluation"""
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    model_name = os.path.basename(ckpt_path)
    features, labels = torch.from_numpy(feats_np), torch.from_numpy(labels_np)
    
    # --- NEW: Extract Epoch or Timestamp for chronological sorting ---
    match = re.search(r'ep(\d+)', model_name)
    sort_val = int(match.group(1)) if match else os.path.getmtime(ckpt_path)
    
    r1, r5, r10, map_val, baks, baus = compute_reid_metrics(features, labels, known_classes, device='cpu', sim_thresh=0.4)
    ari, nmi, discovered_ids = compute_clustering_metrics_leiden(args, feats_np, labels_np, species_preds_np, sim_thresh=0.50, k1=15, lambda_val=0.3)

    return {
        'Model Name': model_name.replace('.pth', ''),
        'Epoch': sort_val, # <--- Added this key
        'Rank-1 (%)': r1, 'Rank-5 (%)': r5, 'Rank-10 (%)': r10, 'mAP (%)': map_val,
        'BaKS': baks, 'BAUS': baus, 'H-Score': np.sqrt(baks * baus), 
        'Baseline ARI': ari, 'Baseline NMI': nmi, 'Baseline IDs': discovered_ids,
        'Species Acc (%)': accuracy_score(species_labels_np, species_preds_np) * 100,    
        'Species BAcc (%)': balanced_accuracy_score(species_labels_np, species_preds_np) * 100,  
        'ckpt_path': ckpt_path 
    }

def tune_model_worker(ckpt_path, feats_np, labels_np, species_preds_np):
    """PHASE 2.5: Exhaustive Parameter Tuning for Top Models"""
    model_name = os.path.basename(ckpt_path).replace('.pth', '')
    best_ari_params, best_id_params = tune_leiden_hyperparameters(feats_np, labels_np, species_preds_np, disable_pbar=True)
    
    return {
        'Model Name': model_name,
        'ARI (Pure)': best_ari_params['ari'], 'NMI (Pure)': best_ari_params['nmi'], 'IDs (Pure)': best_ari_params['discovered_ids'],
        'Thresh (Pure)': best_ari_params['thresh'], 'k1 (Pure)': best_ari_params['k1'], 'Lambda (Pure)': best_ari_params['lambda'],
        'ARI (ID-Match)': best_id_params['ari'], 'NMI (ID-Match)': best_id_params['nmi'], 'IDs (ID-Match)': best_id_params['discovered_ids'],
        'Thresh (ID-Match)': best_id_params['thresh'], 'k1 (ID-Match)': best_id_params['k1'], 'Lambda (ID-Match)': best_id_params['lambda'],
        'ckpt_path': ckpt_path
    }

def generate_tsne_worker(ckpt_path, feats_np, labels_np, args, eval_out_dir):
    """PHASE 3: t-SNE Image Generation"""
    model_name = os.path.basename(ckpt_path).replace('.pth', '')
    features = torch.from_numpy(feats_np)
    
    xy_2d = reduce_to_nd(args, features, n_components=2, method="tsne", seed=42)
    plot_detailed_tsne(xy_2d, labels_np, model_name, os.path.join(eval_out_dir, f'tsne_2d_{model_name}.png'))
    
    xyz_3d = reduce_to_nd(args, features, n_components=3, method="tsne", seed=42)
    plot_detailed_tsne(xyz_3d, labels_np, model_name, os.path.join(eval_out_dir, f'tsne_3d_{model_name}.png'))
    plot_interactive_3d_tsne(xyz_3d, labels_np, model_name, os.path.join(eval_out_dir, f'tsne_3d_interactive_{model_name}.html'))
    return model_name

# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================
def generate_submission_csv(image_ids, predicted_ids, species_preds, species_idx_to_name, output_path="submission.csv", true_datasets=None):
    submission_data = []
    
    for i, (image_id, cluster_id, species_idx) in enumerate(zip(image_ids, predicted_ids, species_preds)):
        # 1. Prefer the EXACT dataset name from the competition test CSV if available
        if true_datasets is not None and pd.notna(true_datasets[i]):
            dataset_name = true_datasets[i]
        # 2. Fallback to the model's mapped prediction if it's missing
        else:
            dataset_name = species_idx_to_name.get(species_idx, f"UnknownDataset_{species_idx}")
            
        submission_data.append({
            "image_id": image_id, 
            "cluster": f"cluster_{dataset_name}_{cluster_id}"
        })
        
    pd.DataFrame(submission_data).to_csv(output_path, index=False)
    print(f"--> Successfully saved AnimalCLEF submission to: {output_path}")

def main(args):
    main_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{main_device.type.upper()}] Starting Asynchronous Pipeline...")
    if args.holdout_dataset != None:
        eval_out_dir = os.path.join(args.checkpoints_dir, "evaluation_results_dataset_holdout")
    else:
        eval_out_dir = os.path.join(args.checkpoints_dir, "evaluation_results")
    csv_path = os.path.join(eval_out_dir, 'benchmark_stats.csv')
    os.makedirs(eval_out_dir, exist_ok=True)
    
    test_df = get_test_samples(args)
    test_samples = list(zip(test_df["path"], test_df["global_label"], test_df["species_label"]))
    print(f"Evaluated Test Set Size: {len(test_samples)} images | TRUE Unique Identities: {len(set(l for _, l, _ in test_samples))}")

    # ====================================================================
    # --- ROUTE 1 - GENERATE ANIMALCLEF SUBMISSION ---
    # ====================================================================
    if args.generate_submission and args.checkpoint_path:
        print(f"\n[ SUBMISSION MODE ] Generating CSV using: {os.path.basename(args.checkpoint_path)}")
        mapping_df = pd.read_csv(os.path.join(args.checkpoints_dir, "pipeline_metadata.csv"), low_memory=False)
        species_idx_to_name = mapping_df.set_index('species_label')['dataset'].to_dict() if 'dataset' in mapping_df.columns else {}
        model, _ = ReIDModel.load(args.checkpoint_path, args=args, device=main_device)
        test_dataset = UniversalGraphDataset(
            samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir, mode=args.data_mode, rebuild_cache=False, 
            img_size=args.img_size, n_hops=args.n_hops, features=args.features, cae_version=args.cae_version, 
            cae_weights_path=args.cae_weights_path, cae_latent_dim=args.cae_latent_dim, felz_scale=args.felz_scale, 
            felz_sigma=args.felz_sigma, min_size=args.felz_min_size, num_bins=args.num_hog_bins
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        feats, _, species_preds, _ = extract_features(model, test_loader, main_device)
        
        # 5. Run Leiden clustering
        print("\n--> Running Leiden Graph Clustering & k-Reciprocal Re-Ranking...")
        predicted_ids = predict_clusters_leiden(
            feats.cpu().numpy(), 
            species_preds.cpu().numpy(), 
            sim_thresh=args.leiden_thresh, 
            k1=args.leiden_k1, 
            lambda_val=args.leiden_lambda
        )
        # Extract the real dataset column (e.g., 'TexasHornedLizards') if it exists in the test CSV
        true_datasets = test_df['dataset'].tolist() if 'dataset' in test_df.columns else None

        generate_submission_csv(
            image_ids=test_df['image_id'].tolist(), 
            predicted_ids=predicted_ids, 
            species_preds=species_preds.cpu().numpy(), 
            species_idx_to_name=species_idx_to_name, 
            output_path=os.path.join(args.checkpoints_dir, "submission.csv"),
            true_datasets=true_datasets
        )
        return 
    # ====================================================================

    gnn_dir = os.path.join(args.checkpoints_dir, "gnn")
    pth_files = glob.glob(os.path.join(gnn_dir, "*.pth"))
    if not pth_files: return print(f"No .pth files found in {gnn_dir}")

    extracted_data = {}

    # --- PHASE 1: SEQUENTIAL GPU INFERENCE ---
    if args.use_existing_csv and os.path.exists(csv_path):
        print(f"\n[ PHASE 1 SKIPPED: Using existing benchmark results ]")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n--- PHASE 1: SEQUENTIAL GPU FEATURE EXTRACTION ---")
        shared_dataset = UniversalGraphDataset(
            samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir, mode=args.data_mode, rebuild_cache=False, 
            img_size=args.img_size, n_hops=args.n_hops, features=args.features, cae_version=args.cae_version, 
            cae_weights_path=args.cae_weights_path, cae_latent_dim=args.cae_latent_dim, 
            seeds_num_superpixels=args.seeds_num_superpixels, seeds_num_levels=args.seeds_num_levels, 
            seeds_prior=args.seeds_prior, seeds_histogram_bins=args.seeds_histogram_bins, num_bins=args.num_hog_bins
        )
        shared_loader = DataLoader(shared_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)

        for ckpt in sorted(pth_files):
            print(f"\nProcessing: {os.path.basename(ckpt)}")
            try:
                model, train_classes = ReIDModel.load(ckpt, args=args, device=main_device)
                features, labels, species_preds, species_labels = extract_features(model, shared_loader, main_device)
                extracted_data[ckpt] = (features.cpu().numpy(), labels.cpu().numpy(), species_preds.cpu().numpy(), species_labels.cpu().numpy(), set(train_classes) if train_classes is not None else set())
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to infer {os.path.basename(ckpt)}: {e}")

        # --- PHASE 2: PARALLEL CPU BASELINE METRICS ---
        print(f"\n--- PHASE 2: PARALLEL METRICS CALCULATION ({args.parallel_workers} Workers) ---")
        spawn_context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=spawn_context) as executor:
            futures = {executor.submit(evaluate_metrics_worker, c, f, l, sp, k, args, sl): c for c, (f, l, sp, sl, k) in extracted_data.items()}
            results = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Baseline Metrics")]
                    
        df = pd.DataFrame(results).sort_values(by='Epoch', ascending=True).reset_index(drop=True)
        
        print("\n================ BENCHMARK SUMMARY ================")
        print(df.drop(columns=['ckpt_path', 'Epoch']).to_string(index=False)) # Hide the epoch col for cleaner terminal output
        plot_benchmark_results(df, eval_out_dir)
        df.to_csv(csv_path, index=False)
        
        # --- PHASE 2.5: TUNE ONLY THE TOP MODELS ---
        top_k_tune = min(args.top_k_tune, len(df))
        print(f"\n--- PHASE 2.5: EXHAUSTIVE LEIDEN TUNING (Top {top_k_tune} Models) ---")
        top_candidates = df.sort_values(by='Baseline ARI', ascending=False).head(top_k_tune)

        with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=spawn_context) as executor:
            tune_futures = {executor.submit(tune_model_worker, row.ckpt_path, *extracted_data[row.ckpt_path][:3]): row.ckpt_path for row in top_candidates.itertuples()}
            tuning_results = [f.result() for f in tqdm(as_completed(tune_futures), total=len(tune_futures), desc="Tuning Top Models")]

        tuning_df = pd.DataFrame(tuning_results).sort_values(by='ARI (ID-Match)', ascending=False).reset_index(drop=True)
        tuning_df.to_csv(os.path.join(eval_out_dir, 'tuning_stats.csv'), index=False)
        
        print("\n================ TUNING SUMMARY ================")
        print(tuning_df.drop(columns=['ckpt_path']).to_string(index=False))
        plot_tuned_clustering(tuning_df, eval_out_dir)

    # --- PHASE 3: PARALLEL TOP-K VISUALIZATION ---
    if args.top_k_detailed > 0:
        top_k = min(args.top_k_detailed, len(df))
        print(f"\n--- PHASE 3: PARALLEL t-SNE GENERATION (Top {top_k} Models) ---")
        
        # FIX: Sort by Baseline ARI since df is from Phase 2
        top_models = df.sort_values(by='Baseline ARI', ascending=False).head(top_k)
        
        if args.use_existing_csv and not extracted_data:
            print("Re-extracting features sequentially for Top-K models...")
            for row in top_models.itertuples():
                checkpoint_data = torch.load(row.ckpt_path, map_location=main_device)
                saved_args = checkpoint_data.get('args', vars(args) if hasattr(args, '__dict__') else args)
                model, train_classes = ReIDModel.load(row.ckpt_path, args=args, device=main_device)
                
                td = UniversalGraphDataset(
                    samples=test_samples, root_dir=args.root_dir, cache_dir=args.cache_dir, mode=args.data_mode, rebuild_cache=False, 
                    img_size=saved_args.get('img_size', args.img_size), felz_scale=saved_args.get('felz_scale', args.felz_scale), 
                    felz_sigma=saved_args.get('felz_sigma', args.felz_sigma), min_size=saved_args.get('felz_min_size', args.felz_min_size), 
                    num_bins=saved_args.get('num_hog_bins', args.num_hog_bins), n_hops=saved_args.get('n_hops', args.n_hops), 
                    features=saved_args.get('features', args.features), cae_version=saved_args.get('cae_version', args.cae_version), 
                    cae_weights_path=saved_args.get('cae_weights_path', args.cae_weights_path), cae_latent_dim=saved_args.get('cae_latent_dim', args.cae_latent_dim)
                )
                tl = DataLoader(td, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                f, l, sp, sl = extract_features(model, tl, main_device)
                extracted_data[row.ckpt_path] = (f.cpu().numpy(), l.cpu().numpy(), sp.cpu().numpy(), sl.cpu().numpy(), set(train_classes) if train_classes is not None else set())
                del model
                torch.cuda.empty_cache()

        spawn_context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.parallel_workers, mp_context=spawn_context) as executor:
            tsne_futures = [executor.submit(generate_tsne_worker, row.ckpt_path, extracted_data[row.ckpt_path][0], extracted_data[row.ckpt_path][1], args, eval_out_dir) for row in top_models.itertuples()]
            for future in tqdm(as_completed(tsne_futures), total=len(tsne_futures), desc="Generating Plots"):
                future.result()

    print(f"\n--> All evaluations complete. Outputs saved to: {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Re-ID Comprehensive Evaluator")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_long_run_v2")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k")
    parser.add_argument("--cache_dir", type=str, default="src/images/reid-10k/graph_cache_pool")
    parser.add_argument("--holdout_dataset", type=str, default=None)
    parser.add_argument("--holdout_species", type=str, default=None)
    parser.add_argument("--eval_filter_species", type=str, default=None)
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "memory", "lazy"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds_num_superpixels", type=int, default=300)
    parser.add_argument("--seeds_num_levels", type=int, default=4)
    parser.add_argument("--seeds_prior", type=int, default=1)
    parser.add_argument("--seeds_histogram_bins", type=int, default=4)
    parser.add_argument("--top_k_detailed", type=int, default=5)
    parser.add_argument("--use_existing_csv", action="store_true")
    parser.add_argument("--parallel_workers", type=int, default=4)
    parser.add_argument("--max_images_per_id", type=int, default=None)
    parser.add_argument("--features", nargs="+", default=["color", "pos", "hog", "lbp", "texture"])
    parser.add_argument("--n_hops", type=int, default=1)
    parser.add_argument("--num_hog_bins", type=int)
    parser.add_argument("--cae_latent_dim", type=int, default=64)
    parser.add_argument("--cae_version", type=str, default="none")
    parser.add_argument("--cae_weights_path", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--generate_submission", action="store_true")
    parser.add_argument("--top_k_tune", type=int, default=10)
    parser.add_argument("--leiden_thresh", type=float, default=0.58)
    parser.add_argument("--leiden_k1", type=int, default=20)
    parser.add_argument("--leiden_lambda", type=float, default=0.2)
    args = parser.parse_args()
    main(args)