import os
import glob
import argparse
import warnings
import re
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*The number of unique classes is greater than 50%.*")
warnings.filterwarnings("ignore", message=".*A single label was found in.*")
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
# Scikit-Learn & Scipy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
# Graph Clustering
import hdbscan
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
# Custom Modules
from train_test_prototype import ReIDModel, reduce_to_nd
from dataloader import UniversalGraphDataset
from torch_geometric.loader import DataLoader
import timm
import torchvision.transforms as T
from types import SimpleNamespace
import concurrent.futures
import traceback

# =====================================================================
# 1. FEATURE EXTRACTION & DATA UTILS
# =====================================================================
from PIL import Image
from torch.utils.data import Dataset

class WildlifePathDataset(Dataset):
    """Wraps a list of image paths into a format wildlife_tools can process."""
    def __init__(self, paths, root_dir):
        self.paths = paths
        self.root_dir = root_dir
        self.root = root_dir
        self.transform = None
        self.metadata = pd.DataFrame({'image_id': paths, 'label': [0] * len(paths)})   
        self.col_label = 'label'
        self.col_image = 'image'
        self.image_size = None
        self.label_to_idx = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        full_path = os.path.join(self.root_dir, self.paths[idx])
        img = Image.open(full_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
class WS():
    def __init__(self, root_dir, gnn_feats=None, path_to_idx=None, mega_cache_path = None):
        self.root_dir = root_dir
        device = 'cuda'
        batch_size = 32

        if mega_cache_path:
            os.makedirs(mega_cache_path, exist_ok=True)
        # MegaDescriptor pipeline (always present)
        self.pipeline_mega = SimilarityPipeline(
            matcher = CosineSimilarity(),
            extractor = DeepFeatures(
                model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(),
                device=device,
                batch_size=batch_size,
                cache_path=mega_cache_path
            ),
            transform = T.Compose([
                T.Resize(size=(384, 384)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
            calibration = None
        )

        # Start pipelines list with Mega only
        self.pipelines = [self.pipeline_mega]

        # Add the current GNN pipeline if features are provided
        if gnn_feats is not None and path_to_idx is not None:
            pipeline_gnn = SimilarityPipeline(
                extractor = PrecomputedExtractor(gnn_feats, path_to_idx),
                matcher = CosineSimilarity(),
                transform = None,
                calibration = None
            )
            self.pipelines.append(pipeline_gnn)

    def apply_ws(self, path_list):
        B = min(1000, len(path_list))
        dataset = WildlifePathDataset(path_list, self.root_dir)
        wildfusion = WildFusion(calibrated_pipelines=self.pipelines, priority_pipeline=self.pipeline_mega)
        similarity = wildfusion(dataset, dataset, B=B)
        return similarity.astype(np.float64)


class PrecomputedExtractor:
    """
    Wraps a pre‑extracted feature tensor (N, D) and a dict mapping path → row index.
    Implements the interface expected by SimilarityPipeline.extractor.
    """
    def __init__(self, features_tensor, path_to_index):
        self.features = features_tensor          
        self.path_to_index = path_to_index

    def __call__(self, dataset):
        paths = dataset.paths
        indices = [self.path_to_index[p] for p in paths]
        feats = self.features[indices].numpy()
        return SimpleNamespace(features=feats)

def get_test_samples(args):
    test_df = pd.read_csv(args.csv_path, low_memory=False)
    print(f"--> Initial CSV loaded. Total rows: {len(test_df)}")

    # --- holdout filters ---
    if args.holdout_dataset:
        if 'dataset' in test_df.columns:
            safe_target = args.holdout_dataset.strip().lower()
            test_df['dataset_safe'] = test_df['dataset'].astype(str).str.strip().str.lower()
            test_df = test_df[test_df['dataset_safe'] == safe_target].reset_index(drop=True)
            if len(test_df) == 0:
                unique_ds = pd.read_csv(args.csv_path, low_memory=False)['dataset'].dropna().unique()
                raise ValueError(f"No images found for dataset '{args.holdout_dataset}'. Available datasets in CSV: {unique_ds}")

    if hasattr(args, 'species') and args.species is not None:
        if 'species' in test_df.columns:
            safe_target = args.species.strip().lower()
            test_df['species_safe'] = test_df['species'].astype(str).str.strip().str.lower()
            test_df = test_df[test_df['species_safe'] == safe_target].reset_index(drop=True)

    if args.holdout_species:
        if 'species' in test_df.columns:
            safe_target = args.holdout_species.strip().lower()
            test_df['species_safe'] = test_df['species'].astype(str).str.strip().str.lower()
            test_df = test_df[test_df['species_safe'] == safe_target].reset_index(drop=True)

    # --- merge with base test CSV if provided (mixed domain) ---
    if getattr(args, 'base_test_csv', None) and os.path.exists(args.base_test_csv):
        base_df = pd.read_csv(args.base_test_csv, low_memory=False)
        test_df = pd.concat([test_df, base_df], ignore_index=True)
        test_df = test_df.drop_duplicates(subset=['path']).reset_index(drop=True)
        print(f"--> Mixed Test Set Size: {len(test_df)} rows.")

    # --- Safe identity handling ---
    if 'identity' not in test_df.columns:
        if 'animal_id' in test_df.columns:
            test_df['identity'] = test_df['animal_id'].astype(str)
        else:
            test_df['identity'] = 'unknown'

    test_df['identity'] = test_df['identity'].fillna('unknown').astype(str)

    # --- Encode labels ---
    from sklearn.preprocessing import LabelEncoder
    test_df['global_label'] = LabelEncoder().fit_transform(test_df['identity'])

    if 'species' in test_df.columns:
        test_df['species'] = test_df['species'].fillna('unknown').astype(str)
        test_df['species_label'] = LabelEncoder().fit_transform(test_df['species'])
    else:
        test_df['species_label'] = 0

    # --- Optional subsampling ---
    if hasattr(args, 'max_images_per_id') and args.max_images_per_id is not None:
        test_df = test_df.groupby('global_label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.max_images_per_id), random_state=42)
        ).reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError("All images were dropped during filtering.")

    return test_df

def get_dataloader(df, args):
    samples = list(zip(df["path"], df["global_label"], df["species_label"]))
    ds = UniversalGraphDataset(
        samples=samples, root_dir=args.root_dir, cache_dir=args.cache_dir, 
        mode=args.data_mode, rebuild_cache=False, 
        img_size=args.img_size, n_hops=args.n_hops, features=args.features, 
        cae_version=args.cae_version, cae_weights_path=args.cae_weights_path, 
        cae_latent_dim=args.cae_latent_dim, seeds_num_superpixels=args.seeds_num_superpixels, 
        seeds_num_levels=args.seeds_num_levels, seeds_prior=args.seeds_prior, 
        seeds_histogram_bins=args.seeds_histogram_bins, num_bins=args.num_hog_bins
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

def extract_features(model, dataloader, device):
    model.eval()
    all_emb, all_labels, all_species_preds, all_species_labels = [], [], [], []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting Features (GNN)", leave=False):
            data = data.to(device)
            y_stacked = data.y.view(-1, 2)
            labels = y_stacked[:, 0]
            species_labels = y_stacked[:, 1]
            
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                out = model(data)
                
            if isinstance(out, tuple):
                emb, species_logits = out
            else:
                emb = out
                species_logits = torch.zeros((emb.size(0), 1), device=device)
                
            emb = F.normalize(emb, p=2, dim=1) 
            probs = F.softmax(species_logits, dim=1)
            _, species_preds = torch.max(probs, dim=1)
            
            all_emb.append(emb.cpu())
            all_species_labels.append(species_labels.cpu())
            all_labels.append(labels.cpu())
            all_species_preds.append(species_preds.cpu())
            
    return torch.cat(all_emb), torch.cat(all_labels), torch.cat(all_species_preds), torch.cat(all_species_labels) 

# =====================================================================
# 2. METRICS
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

def compute_reid_metrics(sim_matrix, labels, known_classes, raw_identities, device='cpu', sim_thresh=0.4):
    sim_matrix = sim_matrix.to(device)
    labels = labels.to(device)
    
    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    sorted_labels = labels[sorted_indices]

    top1_sims = sim_matrix.max(dim=1).values
    top1_labels = sorted_labels[:, 0]
    
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
    
    # Convert predictions back to raw string identities for strict matching
    y_true_raw = raw_identities
    y_pred_raw = np.array([
        raw_identities[match_idx] if sim >= sim_thresh else "-1" 
        for match_idx, sim in zip(sorted_indices[:, 0].cpu().numpy(), top1_sims.cpu().numpy())
    ])
    
    baks = calculate_baks(y_true_raw, y_pred_raw, known_classes)
    baus = calculate_baus(y_true_raw, y_pred_raw, known_classes, unknown_label="-1")
    
    return (cmc_1/valid_queries)*100, (cmc_5/valid_queries)*100, (cmc_10/valid_queries)*100, mAP, baks, baus

# =====================================================================
# 3. GRAPH CLUSTERING (HDBSCAN + Jaccard Reranking)
# =====================================================================
def k_reciprocal_rerank(sim_matrix, k1=20, lambda_value=0.3):
    if isinstance(sim_matrix, np.ndarray):
        sim_matrix = torch.tensor(sim_matrix, dtype=torch.float32)
        
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
    return final_sim.numpy()

def run_hdbscan(similarity_matrix, epsilon=0.50, min_cluster_size=2):
    similarity_matrix[similarity_matrix < 0.15] = 0.0

    distance = (np.max(similarity_matrix) - np.maximum(similarity_matrix, 0)) / (np.max(similarity_matrix) + 1e-8)
    distance = distance.astype(np.float64) 
    
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=2, 
        cluster_selection_epsilon=epsilon, 
        metric='precomputed'
    )
    clusters = clustering.fit(distance)
    
    labels = clusters.labels_.copy()
    max_label = np.max(labels) if len(labels) > 0 and np.max(labels) >= 0 else 0
    for i in range(len(labels)):
        if labels[i] == -1:
            max_label += 1
            labels[i] = max_label
            
    return labels

# =====================================================================
# 4. VISUALIZATION
# =====================================================================
def plot_benchmark_results(results_df, save_dir):
    x = np.arange(len(results_df))
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

    plt.figure(figsize=(14, 7))
    width = 0.35
    plt.bar(x - width/2, results_df['Baseline ARI'], width, label='HDBSCAN ARI', color='#8c564b')
    plt.bar(x + width/2, results_df['Baseline NMI'], width, label='HDBSCAN NMI', color='#e377c2')
    plt.ylabel('Score (0 to 1)', fontsize=12)
    plt.title('Re-ID Model Benchmarks (HDBSCAN Clustering)', fontsize=14)
    plt.xticks(x, results_df['Model Name'], rotation=45, ha='right', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_clustering.png'), dpi=300)
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
# 5. METRICS WORKERS
# =====================================================================
def evaluate_metrics_worker(ckpt_key, fused_sim, labels_np, raw_identities_np, known_classes, species_preds_np, species_labels_np, args):
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    raw_gnn_matrix = torch.tensor(fused_sim, dtype=torch.float32)
    
    # --- NEW: RERANKING TOGGLE ---
    if getattr(args, 'disable_reranking', False):
        reranked_matrix = fused_sim # Bypass Jaccard, pass raw cosine directly to HDBSCAN
    else:
        reranked_matrix = k_reciprocal_rerank(raw_gnn_matrix, k1=args.leiden_k1, lambda_value=args.leiden_lambda)
    
    # -------------------------------------------------------------
    # HDBSCAN Hyperparameter Grid Search
    # -------------------------------------------------------------
    best_eps = 0.50
    best_min_cls = 2
    
    if args.optimize_hdbscan:
        best_ari = -1.0
        best_nmi = -1.0
        best_ids = -1
        
        epsilons = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        min_cluster_sizes = [2, 3, 4]
        
        for eps in epsilons:
            for min_cls in min_cluster_sizes:
                pred_ids = run_hdbscan(reranked_matrix, epsilon=eps, min_cluster_size=min_cls)
                ari = adjusted_rand_score(labels_np, pred_ids)
                
                if ari > best_ari:
                    best_ari = ari
                    best_nmi = normalized_mutual_info_score(labels_np, pred_ids)
                    best_ids = len(set(pred_ids))
                    best_eps = eps
                    best_min_cls = min_cls

        ari = best_ari
        nmi = best_nmi
        discovered_ids = best_ids
        print(f"[{ckpt_key}] Optimized HDBSCAN: eps={best_eps}, min_cls={best_min_cls} (ARI: {ari:.4f})")
    else:
        # Standard hardcoded run
        predicted_ids = run_hdbscan(reranked_matrix, epsilon=0.50, min_cluster_size=2)
        ari = adjusted_rand_score(labels_np, predicted_ids)
        nmi = normalized_mutual_info_score(labels_np, predicted_ids)
        discovered_ids = len(set(predicted_ids))
    
    # Calculate Standard Re-ID Retrieval Metrics
    r1, r5, r10, map_val, baks, baus = compute_reid_metrics(
        raw_gnn_matrix, torch.from_numpy(labels_np), known_classes, raw_identities_np,
        device='cpu', sim_thresh=0.4
    )

    match = re.search(r'ep(\d+)', ckpt_key)
    sort_val = int(match.group(1)) if match else 0

    return {
        'Model Name': ckpt_key, 'Epoch': sort_val,
        'Rank-1 (%)': r1, 'Rank-5 (%)': r5, 'Rank-10 (%)': r10, 'mAP (%)': map_val,
        'BaKS': baks, 'BAUS': baus, 'H-Score': np.sqrt(baks * baus),
        'Baseline ARI': ari, 'Baseline NMI': nmi, 'Baseline IDs': discovered_ids,
        'Opt_Eps': best_eps, 'Opt_MinCls': best_min_cls,
        'Species Acc (%)': accuracy_score(species_labels_np, species_preds_np) * 100,
        'Species BAcc (%)': balanced_accuracy_score(species_labels_np, species_preds_np) * 100,
        'ckpt_path': ckpt_key
    }

def generate_tsne_worker(ckpt_key, feats_np, labels_np, args, eval_out_dir):
    features = torch.from_numpy(feats_np)
    
    xy_2d = reduce_to_nd(args, features, n_components=2, method="tsne", seed=42)
    plot_detailed_tsne(xy_2d, labels_np, ckpt_key, os.path.join(eval_out_dir, f'tsne_2d_{ckpt_key}.png'))
    
    xyz_3d = reduce_to_nd(args, features, n_components=3, method="tsne", seed=42)
    plot_detailed_tsne(xyz_3d, labels_np, ckpt_key, os.path.join(eval_out_dir, f'tsne_3d_{ckpt_key}.png'))
    plot_interactive_3d_tsne(xyz_3d, labels_np, ckpt_key, os.path.join(eval_out_dir, f'tsne_3d_interactive_{ckpt_key}.html'))
    return ckpt_key

# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
def generate_submission_csv(image_ids, predicted_ids, species_preds, species_idx_to_name, output_path="submission.csv", true_datasets=None):
    submission_data = []
    for i, (image_id, cluster_id, species_idx) in enumerate(zip(image_ids, predicted_ids, species_preds)):
        if true_datasets is not None and pd.notna(true_datasets[i]):
            dataset_name = true_datasets[i]
        else:
            dataset_name = species_idx_to_name.get(species_idx, f"UnknownDataset_{species_idx}")
        submission_data.append({"image_id": image_id, "cluster": f"cluster_{dataset_name}_{cluster_id}"})
    pd.DataFrame(submission_data).to_csv(output_path, index=False)
    print(f"--> Successfully saved AnimalCLEF submission to: {output_path}")

def main(args):
    main_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{main_device.type.upper()}] Starting Pipeline...")
    mega_cache_path = os.path.join(args.checkpoints_dir, "feature_cache", "mega")

    args.checkpoints_dir = args.checkpoints_dir.strip()
    args.root_dir = args.root_dir.strip()
    args.csv_path = args.csv_path.strip()
    if args.base_test_csv: args.base_test_csv = args.base_test_csv.strip()
    if args.checkpoint_path: args.checkpoint_path = args.checkpoint_path.strip()
    
    mode_suffix = "_wildfusion" if args.enable_wildfusion else "_gnn_only"
    if args.competition:
        mode_suffix += "_competition"
        
    if getattr(args, 'base_test_csv', None) and args.holdout_dataset != None:
        eval_out_dir = os.path.join(args.checkpoints_dir, f"evaluation_results_mixed_holdout{mode_suffix}{args.eval_suffix}")
    elif args.holdout_dataset != None:
        eval_out_dir = os.path.join(args.checkpoints_dir, f"evaluation_results_dataset_holdout{mode_suffix}{args.eval_suffix}")
    else:
        eval_out_dir = os.path.join(args.checkpoints_dir, f"evaluation_results{mode_suffix}{args.eval_suffix}")
        
    csv_path = os.path.join(eval_out_dir, 'benchmark_stats.csv')
    os.makedirs(eval_out_dir, exist_ok=True)
    
    test_df = get_test_samples(args)

    # ====================================================================
    # --- ROUTE 1 - GENERATE ANIMALCLEF SUBMISSION ---
    # ====================================================================
    if args.generate_submission and args.checkpoint_path:
        print(f"\n[ SUBMISSION MODE ] Generating CSV using: {os.path.basename(args.checkpoint_path)}")
        mapping_path = os.path.join(args.checkpoints_dir, "pipeline_metadata.csv")
        mapping_df = pd.read_csv(mapping_path, low_memory=False) if os.path.exists(mapping_path) else pd.DataFrame()
        species_idx_to_name = mapping_df.set_index('species_label')['dataset'].to_dict() if 'dataset' in mapping_df.columns else {}
        
        model_gnn, _ = ReIDModel.load(args.checkpoint_path, args=args, device=main_device)
        test_loader = get_dataloader(test_df, args)
        
        print("--> Running Standard GNN Feature Extraction...")
        gnn_feats, _, species_preds, _ = extract_features(model_gnn, test_loader, main_device)
        
        print("--> Running Similarity & HDBSCAN Clustering...")
        gnn_sim_matrix = torch.mm(gnn_feats, gnn_feats.t())
        reranked_matrix = k_reciprocal_rerank(gnn_sim_matrix, k1=args.leiden_k1, lambda_value=args.leiden_lambda)
        
        predicted_ids = run_hdbscan(reranked_matrix, epsilon=0.50, min_cluster_size=4)
            
        true_datasets = test_df['dataset'].tolist() if 'dataset' in test_df.columns else None
        generate_submission_csv(
            image_ids=test_df['image_id'].tolist(), predicted_ids=predicted_ids, species_preds=species_preds.numpy(), 
            species_idx_to_name=species_idx_to_name, output_path=os.path.join(args.checkpoints_dir, "submission.csv"), true_datasets=true_datasets
        )
        return

    # ====================================================================
    # --- ROUTE 2 - METRICS EVALUATION ---
    # ====================================================================
    gnn_dir = os.path.join(args.checkpoints_dir, "gnn")
    pth_files = glob.glob(os.path.join(gnn_dir, "*.pth"))
    if not pth_files: return print(f"No .pth files found in {gnn_dir}")

    extracted_data = {}
    
    if args.use_existing_csv and os.path.exists(csv_path):
        print(f"\n[ PHASE 1 SKIPPED: Using existing benchmark results ]")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n--- PHASE 1: SEQUENTIAL GPU FEATURE EXTRACTION ---")
        test_loader = get_dataloader(test_df, args)

        for ckpt in sorted(pth_files):
            print(f"\nProcessing Checkpoint: {os.path.basename(ckpt)}")
            try:
                model, train_classes = ReIDModel.load(ckpt, args=args, device=main_device)

                gnn_query_feats, _, species_preds, species_labels = extract_features(model, test_loader, main_device)
                path_list = test_df['path'].tolist()
                path_to_idx = {p: i for i, p in enumerate(path_list)}
                dict_key = os.path.basename(ckpt).replace('.pth', '')
                
                extracted_data[dict_key] = {
                    'gnn_query': gnn_query_feats,
                    'train_classes': set(train_classes) if train_classes is not None else set(),
                    'species_preds': species_preds.numpy(),
                    'species_labels': species_labels.numpy(),
                    'path_to_idx': path_to_idx, 
                }
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to infer {os.path.basename(ckpt)}: {e}")

        # --- PHASE 2: PARALLEL METRICS CALCULATION ---
        print(f"\n--- PHASE 2: PARALLEL METRICS CALCULATION ---")
        
        if not extracted_data:
            print("[CRITICAL ERROR] No models were successfully processed in Phase 1!")
            print("Scroll up to see the 'Failed to infer' error messages.")
            return
        
        path_list = test_df['path'].tolist()
        base_path_to_idx = {p: i for i, p in enumerate(path_list)}

        task_payloads = []
        for k, model_data in extracted_data.items():
            gnn_feats = model_data['gnn_query']

            if args.enable_wildfusion:
                ws = WS(root_dir=args.root_dir,
                        gnn_feats=gnn_feats,
                        path_to_idx=base_path_to_idx,
                        mega_cache_path=mega_cache_path)
                fused_sim = ws.apply_ws(path_list)
            else:
                gnn_sim = torch.mm(gnn_feats, gnn_feats.t())
                fused_sim = gnn_sim.numpy().astype(np.float64)

            task_payloads.append({
                'ckpt_key': k,
                'fused_sim': fused_sim,
                'labels_np': test_df['global_label'].values,
                'raw_identities_np': test_df['identity'].values,
                'known_classes': model_data['train_classes'],
                'species_preds_np': model_data['species_preds'],
                'species_labels_np': model_data['species_labels'],
                'args': args
            })

        results = []
        # Cap thread workers to prevent over-subscription (HDBSCAN is multi-threaded in C)
        max_workers = min(args.parallel_workers, 8) if args.parallel_workers > 0 else None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_metrics_worker, **payload) for payload in task_payloads]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing Metrics (Parallel)"):
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"A worker generated an exception: {exc}")
                    traceback.print_exc()
                    
        df = pd.DataFrame(results)
        
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            old_df = old_df[~old_df['Model Name'].isin(df['Model Name'])]
            df = pd.concat([old_df, df])
            
        df = df.sort_values(by=['Epoch', 'Model Name'], ascending=[True, True]).reset_index(drop=True)
        
        print("\n================ BENCHMARK SUMMARY ================")
        print(df.drop(columns=['ckpt_path', 'Epoch']).to_string(index=False))
        plot_benchmark_results(df, eval_out_dir)
        df.to_csv(csv_path, index=False)

        # --- PHASE 3: SEQUENTIAL TOP-K VISUALIZATION ---
        if args.top_k_detailed > 0:
            top_k = min(args.top_k_detailed, len(df))
            print(f"\n--- PHASE 3: SEQUENTIAL t-SNE GENERATION (Top {top_k} Models) ---")
            
            top_models = df.sort_values(by='Baseline ARI', ascending=False).head(top_k)
            for _, row in tqdm(top_models.iterrows(), total=len(top_models), desc="Generating Plots"):
                m_name = row['Model Name']
                final_feats = extracted_data[m_name]['gnn_query'].numpy()
                labels_np = test_df['global_label'].values
                generate_tsne_worker(m_name, final_feats, labels_np, args, eval_out_dir)

    print(f"\n--> All evaluations complete. Outputs saved to: {eval_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Re-ID Comprehensive Evaluator (GNN Only)")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_long_run_v2")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k")
    parser.add_argument("--cache_dir", type=str, default="src/images/reid-10k/graph_cache_pool")
    parser.add_argument("--holdout_dataset", type=str, default=None)
    parser.add_argument("--holdout_species", type=str, default=None)
    parser.add_argument("--eval_filter_species", type=str, default=None)
    parser.add_argument("--species", type=str, default=None, help="Filter the test set to evaluate only a specific species (e.g., 'lynx')")
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "memory", "lazy"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
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
    
    parser.add_argument("--leiden_thresh", type=float, default=0.58)
    parser.add_argument("--leiden_k1", type=int, default=12)
    parser.add_argument("--leiden_lambda", type=float, default=0.2)
    
    parser.add_argument("--base_test_csv", type=str, default=None)

    parser.add_argument("--enable_wildfusion", action="store_true", help="Fuse MegaDescriptor with GNN instead of GNN-only")
    parser.add_argument("--competition", action="store_true", help="Evaluate on competition test set (separate output folder)")
    parser.add_argument("--eval_suffix", type=str, default="", help="Optional suffix appended to the eval output directory.")
    parser.add_argument("--optimize_hdbscan", action="store_true", help="Run hyperparameter search for HDBSCAN instead of using hardcoded values")
    parser.add_argument("--disable_reranking", action="store_true", help="Skip Jaccard k-reciprocal reranking and use raw cosine similarity")
    args = parser.parse_args()
    main(args)