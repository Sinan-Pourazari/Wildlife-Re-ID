import datetime
import os
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
import random
import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import embedding_clusterings as ec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from loss_mining_tools import batch_hard_triplet_loss, PKBatchSampler, batch_semi_hard_triplet_loss, batch_compactness_loss, batch_topk_triplet_loss
from PIL import Image
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from gnn.gnn import image_to_superpixel_graph, GNNEncoder
import torch.nn.functional as F
from dataloader import InMemoryGraphDataset, UniversalGraphDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


print(f"Using {device} device")
class SimpleDataset(TorchDataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device) #TODO REM
        self.labels   = torch.tensor(labels, dtype=torch.long).to(device) #TODO REM

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TripletDataset(TorchDataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)

        # Precompute indices for each label (so positives are easy)
        self.label_to_indices = {}
        for i, y in enumerate(self.labels.tolist()):
            self.label_to_indices.setdefault(y, []).append(i)

        self.unique_labels = list(self.label_to_indices.keys())

        # Safety: Triplet sampling requires at least 2 labels and >=2 samples per label
        assert len(self.unique_labels) >= 2, "TripletDataset needs at least 2 different labels."
        assert all(len(idxs) >= 2 for idxs in self.label_to_indices.values()), \
            "Every label in TripletDataset must have at least 2 samples."

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor = self.features[idx]
        label  = int(self.labels[idx].item())

        # ----- Positive: same label, different index
        pos_candidates = self.label_to_indices[label]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(pos_candidates)
        positive = self.features[pos_idx]

        # ----- Negative: pick a different label, then a random index from it
        neg_label = label
        while neg_label == label:
            neg_label = random.choice(self.unique_labels)
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative = self.features[neg_idx]

        return anchor, positive, negative
    
class GraphImageDataset(PyGDataset):
    def __init__(self, samples, root_dir, n_segments=50):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.n_segments = n_segments

    def len(self):
        return len(self.samples)

    def get(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.root_dir, filename)
        img = Image.open(image_path).convert("RGB")
        #img = img.resize((256, 256))  # or 

        graph = image_to_superpixel_graph(img, n_segments=self.n_segments)
        graph.y = torch.tensor([int(label)], dtype=torch.long)
        return graph
    
class ReIDModel(nn.Module):
    def __init__(self, num_classes, in_dim=14, hidden_dim=256, gnn_out_dim=256, emb_dim=512, use_hybrid_pooling=True):
        super().__init__()
        self.save_params = {
            'num_classes': num_classes,
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'gnn_out_dim': gnn_out_dim,
            'emb_dim': emb_dim,
            'use_hybrid_pooling': use_hybrid_pooling
        }
        # Initialize the GNN Encoder with provided params
        self.encoder = GNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=gnn_out_dim)
        
        # Calculate the actual size coming out of the GNN
        # If pooling Mean + Max, the dimension is doubled
        self.gnn_feature_size = gnn_out_dim * 2 if use_hybrid_pooling else gnn_out_dim

        self.head = nn.Sequential(
            nn.Linear(self.gnn_feature_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Added for training stability at 140k scale
            nn.GELU(),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, emb_dim),
        )

        # Classification head (Used ONLY during training for CE loss)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, data):
        # 1. Extract graph-level features
        z = self.encoder(data) 
        
        # 2. Project to Re-ID embedding space
        features = self.head(z)
        
        # 3. L2 Normalize for Cosine Similarity / Metric Learning
        embeddings = F.normalize(features, p=2, dim=1)

        if self.training:
            # Return both for the dual-loss training loop
            logits = self.classifier(features) # Use un-normalized features for CE
            return embeddings, logits ,features
        
        return embeddings
    
    # --- CHANGE 1: Accept train_classes instead of label_encoder ---
    def save(self, args, epoch, optimizer, train_classes=None):
        """Saves weights, metadata, and hyperparameters with attribute safety."""
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # --- DEFENSIVE CHECK ---
        if hasattr(self, 'save_params'):
            hyperparams = self.save_params
        else:
            print("Warning: 'save_params' not found in model. Using default fallback.")
            hyperparams = {
                'in_dim': 14,
                'hidden_dim': 512,
                'gnn_out_dim': 256,
                'emb_dim': 512,
                'use_hybrid_pooling': True
            }

        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        species = getattr(args, 'holdout_species', 'universal') or 'universal'
        model_name = f"gnn_reid_{species}_{timestamp}.pth"
        save_path = os.path.join(args.checkpoint_dir, model_name)

        # Prepare payload
        payload = {
            'model_state_dict': self.state_dict(),
            'hyperparameters': hyperparams,
            'optimizer_state_dict': optimizer.state_dict(), 
            'epoch': epoch,                                 
            'args': vars(args) if hasattr(args, '__dict__') else args,
            # --- CHANGE 2: Save the raw list directly ---
            'label_encoder_classes': train_classes, 
            'timestamp': timestamp
        }
    
        print(f"Saving model to {save_path}...")
        torch.save(payload, save_path)
        print(f"--> Save complete.")
        return save_path

    @staticmethod
    def load(checkpoint_path, device='cpu'):
        """Reconstructs the model, handling both old and new checkpoint formats."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        hyperparams = checkpoint.get('hyperparameters', {})
        if not hyperparams:
            hyperparams = {
                'in_dim': checkpoint.get('in_dim', 14),
                'hidden_dim': checkpoint.get('hidden_dim', 512),
                'gnn_out_dim': checkpoint.get('gnn_out_dim', 256),
                'emb_dim': checkpoint.get('emb_dim', 512)
            }

        if 'num_classes' not in hyperparams:
            hyperparams['num_classes'] = 1

        model = ReIDModel(**hyperparams)
        
        # FILTER OUT THE CLASSIFIER
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
        
        # Load the filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        return model, checkpoint.get('label_encoder_classes')

def train_one_epoch(loader, model, optimizer, margin=1.0):
    model.train()
    total = 0.0
    total_triplet = 0.0
    total_ce = 0.0
    total_compact = 0.0
    criterion_ce = nn.CrossEntropyLoss()
    for data in loader:
        data = data.to(device)
        labels = data.y.view(-1).to(device)

        # Unpack the two outputs
        emb, logits, features = model(data)

        # Calculate losses
        #loss_triplet = batch_hard_triplet_loss(emb, labels, margin=margin)
        loss_triplet = batch_topk_triplet_loss(emb, labels, margin=margin, k_pos=1, k_neg=10)
        loss_ce = 0.25 * criterion_ce(logits, labels)

        
        loss_compact = 6 * batch_compactness_loss(features, labels)

        # Combined Loss
        loss = loss_triplet + loss_ce + loss_compact

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        total_triplet += loss_triplet
        total_ce += loss_ce
        total_compact += loss_compact

    return total / len(loader), total_triplet / len(loader), total_ce / len(loader), total_compact / len(loader) 


def train(loader, model, optimizer, num_epochs, start_epoch=0, args=None, train_classes = None):
    for i in range(start_epoch, num_epochs):
        total_batchloss, triplet, ce, compact = train_one_epoch(loader, model, optimizer, margin=1)
        print(f"epoch {i} batchloss: {total_batchloss}, triplet loss: {triplet}, cross entropy loss: {ce}, compactness loss: {compact}")
        if i % 2 ==0:
            _=model.save(args,epoch=i, optimizer=optimizer, train_classes=train_classes )


def knn_accuracy(embeddings, labels, k=4):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    correct = 0
    for i in range(len(labels)):
        neighbor_labels = labels[indices[i][1:]]  # skip self
        if labels[i] in neighbor_labels:
            correct += 1

    return correct / len(labels)


def eval(model, loader, closed_set: bool):
    from sklearn.metrics.pairwise import cosine_distances
    model.eval()
    emb_list, label_list = [], []

    # 1. Extract Embeddings
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb = model(data)
            emb_list.append(emb)
            label_list.append(data.y.view(-1))

    embeddings = torch.cat(emb_list).cpu().numpy()
    labels = torch.cat(label_list).cpu().numpy()

    # 2. Compute Distance Matrix (Cosine Distance)
    # dists[i, j] is the distance between embedding i and embedding j
    dists = cosine_distances(embeddings)

    # 3. Calculate Rank-1 and mAP
    all_ap = []
    rank1_correct = 0
    num_queries = len(labels)

    for i in range(num_queries):
        query_label = labels[i]
        
        # Get distances for this query, excluding the query itself
        query_dists = dists[i]
        # We want to ignore the distance to itself (which is 0)
        # We do this by setting its distance to infinity
        query_dists[i] = np.inf 
        
        # Sort indices by distance (ascending)
        sorted_indices = np.argsort(query_dists)
        sorted_labels = labels[sorted_indices]

        # --- Rank-1 ---
        if sorted_labels[0] == query_label:
            rank1_correct += 1

        # --- Average Precision (AP) ---
        # Find positions of all matching labels
        matches = (sorted_labels == query_label)
        num_rel = np.sum(matches) # Total number of relevant items in the gallery
        
        if num_rel == 0:
            continue

        # Cumulative sum of matches to get "hits at rank k"
        hits_at_k = np.cumsum(matches)
        # Ranks at which matches occurred (1-indexed)
        ranks = np.arange(1, len(sorted_labels) + 1)
        
        # Precision at each hit: (number of hits) / (current rank)
        precisions = (hits_at_k / ranks) * matches
        
        # AP is the average of precisions at the points where a match was found
        ap = np.sum(precisions) / num_rel
        all_ap.append(ap)

    rank1 = (rank1_correct / num_queries) * 100
    mAP = np.mean(all_ap) * 100

    # UI Helpers
    COLOR = accuracy_to_color(rank1)
    RESET = "\033[0m"
    mode_str = "Closed-Set" if closed_set else "Open-Set"

    print(f"\n[{mode_str} Results]")
    print(f"Rank-1 Accuracy: {COLOR}{rank1:.2f}%{RESET}")
    print(f"mAP:             {COLOR}{mAP:.2f}%{RESET}")

    # 4. Open-Set Specific Clustering Metrics
    if not closed_set:
        memory = ec.IdentityMemory(threshold=0.70, max_exemplars_per_identity=10)
        predicted_ids = [memory.upsert(torch.from_numpy(e))[0] for e in embeddings]
        
        ari = adjusted_rand_score(labels, predicted_ids)
        nmi = normalized_mutual_info_score(labels, predicted_ids)

        print(f"Discovered IDs:  {len(memory.memory)}")
        print(f"ARI:             {ari:.4f}")
        print(f"NMI:             {nmi:.4f}")

    return rank1, mAP

def accuracy_to_color(acc_percent: float) -> str:
    """
    Maps accuracy in [0, 100] to an ANSI RGB color.
    0%   -> red
    100% -> dark green
    """
    acc = max(0.0, min(100.0, acc_percent)) / 100.0

    r = int(255 * (1 - acc))
    g = int(160 * acc)
    b = 0

    return f"\033[38;2;{r};{g};{b}m"



def main(args):
    # Setup Paths
    csv_path = "src/images/reid-10k/metadata.csv"
    img_root = "src/images/reid-10k"  # Base directory where dataset folders live
    cache_pool = "src/images/reid-10k/graph_cache_pool"
    # 1. Load the Universal Metadata
    print("\n[ Loading Metadata ]")
    df = pd.read_csv(csv_path)
    
    # --- FILTER BY SPECIFIC SPECIES ---
    if args.species:
        print(f"--> Filtering dataset to ONLY include: {args.species}")
        df = df[df['species'].isin(args.species)].reset_index(drop=True)
        
        # Safety check to prevent crashing later if typos were made
        if len(df) == 0:
            raise ValueError(f"No images found for the specified species: {args.species}. Check your spelling.")

    # Filter out entries with no cluster_id/identity if necessary
    df['global_identity'] = df['dataset'] + "_" + df['identity'].astype(str)
    
    counts = df['global_identity'].value_counts()
    keep_ids = counts[counts > 1].index
    df = df[df['global_identity'].isin(keep_ids)].reset_index(drop=True)
    # Filter out entries with no cluster_id/identity if necessary
    df['global_identity'] = df['dataset'] + "_" + df['identity'].astype(str)
    
    counts = df['global_identity'].value_counts()
    keep_ids = counts[counts > 1].index
    df = df[df['global_identity'].isin(keep_ids)].reset_index(drop=True)

    # --- NEW: SUBSET LOGIC ---
    if args.subset_fraction < 1.0:
        print(f"\n[ Applying {args.subset_fraction * 100:.0f}% Subset ]")
        unique_ids = df['global_identity'].unique()
        keep_n = max(2, int(len(unique_ids) * args.subset_fraction)) # Keep at least 2 IDs
        
        # Randomly select a subset of identities
        np.random.seed(42) # Keep it reproducible so you test on the same subset
        sampled_ids = np.random.choice(unique_ids, keep_n, replace=False)
        
        # Filter the dataframe to only include those identities
        df = df[df['global_identity'].isin(sampled_ids)].reset_index(drop=True)
        print(f"--> Shrunk dataset to {keep_n} unique identities ({len(df)} total images)")

    le = LabelEncoder()
    df['global_label'] = le.fit_transform(df['global_identity'])

# 3. Apply Holdout Logic
    if args.holdout_species:
        print(f"--> HOLDING OUT SPECIES: {args.holdout_species}")
        train_df = df[df['species'] != args.holdout_species].reset_index(drop=True)
        test_df = df[df['species'] == args.holdout_species].reset_index(drop=True)
        
    elif args.holdout_dataset:
        print(f"--> HOLDING OUT DATASET: {args.holdout_dataset}")
        train_df = df[df['dataset'] != args.holdout_dataset].reset_index(drop=True)
        test_df = df[df['dataset'] == args.holdout_dataset].reset_index(drop=True)
        
    else:
        # Standard Open-Set Split on the whole universe
        print("--> Standard Open-Set Split (No Holdout)")
        unique_labels = df["global_label"].unique()
        train_labels, test_labels = train_test_split(unique_labels, test_size=0.1, random_state=42)
        train_df = df[df["global_label"].isin(train_labels)].reset_index(drop=True)
        test_df = df[df["global_label"].isin(test_labels)].reset_index(drop=True)
        os.mkdir(args.checkpoint_dir)
        test_df.to_csv(f"{args.checkpoint_dir}/current_test_split.csv", index=False)

    
    # No matter how the split was made, we must ensure training labels 
    # are strictly 0 to (N-1) for the Cross Entropy classifier.
    print("\n[ Processing Labels ]")
    train_le = LabelEncoder()
    train_df['contiguous_label'] = train_le.fit_transform(train_df['global_label'])
    
    # Calculate the number of classes for the model init
    num_train_classes = len(train_le.classes_)
    print(f"--> Active Training Classes: {num_train_classes}")

    # 4. Create Sample Lists (mapping path -> label)
    # Train uses the NEW contiguous labels
    train_samples = list(zip(train_df["path"], train_df["contiguous_label"]))
    
    # Test uses the global_labels (Evaluation and Triplet loss don't care about gaps)
    test_samples = list(zip(test_df["path"], test_df["global_label"]))

    print(f"Train size: {len(train_samples)} images | Test size: {len(test_samples)} images")
    # --- Initialize Universal Datasets ---
    print("\n[ Preparing Training Data ]")
    train_dataset = UniversalGraphDataset(
        num_train_classes = num_train_classes,
        samples=train_samples, 
        root_dir=img_root, 
        cache_dir=cache_pool, 
        mode=args.data_mode,
        n_segments=args.segments,
        img_size= args.img_size,
        rebuild_cache=args.rebuild,
        features=args.features
    )

    # Look at the very first graph in the dataset to see how wide the features are
    first_graph = train_dataset[0]
    dynamic_in_dim = first_graph.x.shape[1]
    print(f"--> Dynamically detected Node Feature Dimension (in_dim): {dynamic_in_dim}")
    print("\n[ Preparing Test/Holdout Data ]")

    test_dataset = UniversalGraphDataset(
        num_train_classes = num_train_classes,
        samples=test_samples, 
        root_dir=img_root, 
        cache_dir=cache_pool, 
        mode=args.data_mode, 
        n_segments=args.segments,
        features= args.features
        )

    # DataLoaders
    batch_sampler = PKBatchSampler(train_df["global_label"].values, P=64, K=2)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.workers)

    # Model & Optimizer
    # Look at the first graph to find out the feature width dynamically
    first_graph = train_dataset[0]
    dynamic_in_dim = first_graph.x.shape[1]
    print(f"--> Dynamically detected Node Feature Dimension (in_dim): {dynamic_in_dim}")

    # Model & Optimizer (Notice in_dim is now dynamic)
    model = ReIDModel(num_classes=num_train_classes, in_dim=dynamic_in_dim, hidden_dim=512, gnn_out_dim=256, emb_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- RESUME LOGIC ---
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n[ Resuming Training from: {args.resume} ]")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 1. Load the full model weights (including classifier)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. Load the optimizer's momentum buffers
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # 3. Set the starting epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1 # Start on the *next* epoch
                
            print(f"--> Successfully loaded. Resuming at Epoch {start_epoch}...")
        else:
            print(f"WARNING: No checkpoint found at '{args.resume}'. Starting from scratch.")

    # Train & Evaluate
    # Pass the start_epoch and args into the train loop
    train(train_loader, model, optimizer, num_epochs=args.epochs, start_epoch=start_epoch, args=args, train_classes=train_labels.tolist())
    # --- Saving the Results ---
    print("\n[ Saving Model ]")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Create a unique name based on the holdout or timestamp
    print(f"Saving model...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"gnn_reid_{args.holdout_species or 'universal'}_{timestamp}.pth"
    save_path = os.path.join(save_dir, model_name)

    print(f"--> Model and metadata saved to: {save_path}")
    # Evaluate as an Open Set since the holdout data contains unseen IDs
    print(f"\nEvaluating on Holdout Set...")
    #eval(model, test_loader, closed_set=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildlife Re-ID GNN Trainer")
    
    # Task settings
    parser.add_argument("--closed_set", action="store_true", help="Run in closed-set mode (default is open-set)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    # Dataset scaling settings
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "memory", "lazy"], help="How to load graphs. 'auto' chooses based on dataset size.")
    parser.add_argument("--segments", type=int, default=300, help="Number of superpixels (SLIC segments)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of graph cache (ignore existing .pt files)")
    parser.add_argument("--img_size", type=int, default=1024, help="Max dimension (width or height) for images before graph creation")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of identities to keep (e.g., 0.1 for 10%)")
    parser.add_argument("--species", type=str, nargs="+", default=None, help="List of specific species to use (e.g., --species tiger fox wolf)")
    # Hardware settings
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU workers for DataLoader")
    
    # Holdout settings
    parser.add_argument("--holdout_dataset", type=str, default=None, help="Name of the dataset to hold out for testing (e.g., 'ATRW')")
    parser.add_argument("--holdout_species", type=str, default=None, help="Name of the species to hold out for testing (e.g., 'tiger')")
    
    # checkpoint dir
    parser.add_argument("--checkpoint_dir", type= str, default="checkpoints_long_run_v3", help="name of directory for checkpointing")

    # Resume training flag
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth file to resume training")

    # Feature extractor settings:
    parser.add_argument("--features", nargs="+", default=["color", "pos", "hog", "lbp", "texture"], help="List of node features to extract (color pos hog lbp texture)")
    args = parser.parse_args()
    
    main(args)




#################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_to_nd(
    args,
    embeddings: torch.Tensor,
    n_components: int = 2,
    method: str = "tsne",
    seed: int = 42,

) -> np.ndarray:
    """
    embeddings: torch.Tensor on CPU (N, D)
    returns: np.ndarray (N, 2)
    """
    X = embeddings.numpy()
    n, d = X.shape

    # If too few points, t-SNE isn't stable/valid -> PCA fallback
    if method.lower() == "tsne" and n < 10:
        method = "pca"

    if method.lower() == "pca":
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)

    if method.lower() == "tsne":
        # Standard trick: PCA -> t-SNE for speed/stability
        pca_dim = min(50, d, max(2, n - 1))
        Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

        # perplexity must be < n; choose a safe value
        # typical range: 5..30
        perplexity = min(30, max(5, (n - 1) // 3))
        perplexity = min(perplexity, n - 1)

        tsne = TSNE(
            n_components=n_components,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
            n_jobs=args.workers
        )
        return tsne.fit_transform(Xp)

    raise ValueError(f"Unknown method='{method}'. Use 'pca' or 'tsne'.")


def plot_embedding_2d(
    xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    max_points: int = 5000,
    alpha: float = 0.75,
    s: float = 10.0,
    annotate_top_k: int = 15,
) -> None:
    """
    Scatter plot of 2D embedding with colors by label.
    Optionally annotates centroids of the top-k most frequent labels.
    """
    assert xy.shape[1] == 2

    labels = labels.astype(int)
    n = len(labels)

    # Optional subsampling for speed/readability
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_points, replace=False)
        xy_plot = xy[idx]
        labels_plot = labels[idx]
    else:
        xy_plot = xy
        labels_plot = labels

    unique = np.unique(labels_plot)
    # map labels -> 0..C-1 for coloring
    label_to_idx = {lab: i for i, lab in enumerate(unique)}
    c = np.array([label_to_idx[lab] for lab in labels_plot], dtype=int)

    plt.figure(figsize=(10, 8))
    plt.scatter(xy_plot[:, 0], xy_plot[:, 1], c=c, s=s, alpha=alpha)
    plt.title(f"{title} | N={len(labels_plot)} | classes={len(unique)}")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()

    # Annotate centroids for the most frequent labels (helps sanity-check clustering)
    if annotate_top_k > 0 and len(unique) > 1:
        # counts on the plotted subset
        vals, counts = np.unique(labels_plot, return_counts=True)
        top = vals[np.argsort(-counts)][:annotate_top_k]

        for lab in top:
            mask = labels_plot == lab
            cx, cy = xy_plot[mask].mean(axis=0)
            plt.text(cx, cy, str(lab), fontsize=9)

    plt.show()

#TODO for IDs with only one image, add on the fly rdm iamge argumentation for positive pairs, should be fine with batch pre fetch enabled
#TODO add another Linear layer after the last to give the seperator head a chance to repopulate the dorpout neurons
#TODO indenity aware pk triplet mining ( long taile dists.)