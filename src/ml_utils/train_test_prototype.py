import datetime
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
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
from loss_mining_tools import batch_hard_triplet_loss, PKBatchSampler, batch_semi_hard_triplet_loss, batch_compactness_loss, batch_topk_triplet_loss, batch_topk_semi_hard_triplet_loss
from PIL import Image
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from gnn.gnn import image_to_superpixel_graph, GNNEncoder
import torch.nn.functional as F
from dataloader import InMemoryGraphDataset, UniversalGraphDataset, prepare_augmented_training_data
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
from  pytorch_metric_learning.losses import ArcFaceLoss 
from gnn.cae import train_and_save_cae
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import bitsandbytes as bnb
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
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels   = torch.tensor(labels, dtype=torch.long).to(device) 

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
    def __init__(self,features, cae_latent_dim, num_classes, num_species, in_dim, num_hog_bins \
        , hidden_dim=256, gnn_out_dim=256, emb_dim=512, use_hybrid_pooling=True, edge_strategy="spatial", k_neighbors=5):
        super().__init__()
        self.save_params = {
            'features': features,
            'num_classes': num_classes,
            'cae_latent_dim': cae_latent_dim,
            'num_species': num_species,
            'in_dim': in_dim,
            'hidden_dim': hidden_dim,
            'gnn_out_dim': gnn_out_dim,
            'emb_dim': emb_dim,
            'use_hybrid_pooling': use_hybrid_pooling,
            'edge_strategy': edge_strategy,
            'k_neighbors': k_neighbors,
            'num_hog_bins': num_hog_bins
        }
        # Initialize the GNN Encoder with provided params
        self.encoder = GNNEncoder(num_hog_bins = num_hog_bins,features=features,cae_latent_dim=cae_latent_dim, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=gnn_out_dim, edge_strategy=edge_strategy, k_neighbors=k_neighbors)
        # Calculate the actual size coming out of the GNN
        # If pooling Mean + Max, the dimension is doubled
        #TODO check this see gnn
        #self.gnn_feature_size = gnn_out_dim * 2 if use_hybrid_pooling else gnn_out_dim
        self.gnn_feature_size = hidden_dim * 2 if use_hybrid_pooling else hidden_dim

        self.head = nn.Sequential(
            nn.Linear(self.gnn_feature_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Added for training stability at 140k scale
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, emb_dim),
        )

        # --- BNNeck ---
        self.bottleneck = nn.BatchNorm1d(emb_dim)
        self.bottleneck.bias.requires_grad_(False) # No bias shift

        # ID Classification head (Used ONLY during training for CE loss)
        #self.classifier = nn.Linear(emb_dim, num_classes)

        # Species Classification head
        self.species_classifier = nn.Linear(emb_dim, num_species)

    def forward(self, data):
        # 1. Extract graph-level features
        z = self.encoder(data) 
        
        # 2. Project to Re-ID embedding space
        features = self.head(z)
        bn_features = self.bottleneck(features)
        # 3. L2 Normalize for Cosine Similarity / Metric Learning
        embeddings = F.normalize(bn_features, p=2, dim=1)

        if self.training:
            # Return both for the dual-loss training loop
            #logits = self.classifier(features) # Use un-normalized features for CE
            if self.species_classifier is not None:
                species_logits = self.species_classifier(features)
                return embeddings,  species_logits ,features
            
            return embeddings
        
        elif self.species_classifier is not None:
            species_logits = self.species_classifier(features)
            return embeddings, species_logits

        return embeddings
    
    # --- Accept train_classes instead of label_encoder ---
    # TODO save relevant args parts
    def save(self, args, epoch, optimizer, train_classes=None, species_classes = None):
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            'label_encoder_classes': train_classes,
            'species_classes': species_classes, 
            'timestamp': timestamp
        }
    
        print(f"Saving model to {save_path}...")
        torch.save(payload, save_path)
        print(f"--> Save complete.")
        return save_path

    @staticmethod
    def load(checkpoint_path, args, device='cpu'):
        """Reconstructs the model, handling both old and new checkpoint formats."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        hyperparams = checkpoint.get('hyperparameters', {})
        if not hyperparams:
            raise ValueError(f"Checkpoint {checkpoint_path} is missing 'hyperparameters'. It is too old to be loaded.")
        # Safe fallbacks for older models that didn't save these parameters
        hyperparams['features'] = hyperparams.get('features', args.features)
        hyperparams['cae_latent_dim'] = hyperparams.get('cae_latent_dim', args.cae_latent_dim)
        hyperparams['edge_strategy'] = hyperparams.get('edge_strategy', 'spatial')
        hyperparams['k_neighbors'] = hyperparams.get('k_neighbors', 10)
        hyperparams['num_hog_bins'] = hyperparams.get('num_hog_bins', getattr(args, 'num_hog_bins', 9)) # <--- ADD THIS
        if 'num_classes' not in hyperparams:
            hyperparams['num_classes'] = 1
            
        # Species handling
        species_classes = checkpoint.get('species_classes', None)
        hyperparams['num_species'] = len(species_classes) if species_classes is not None else 1

        # Instantiate the model with the exact hyperparams it was trained with
        model = ReIDModel(**hyperparams) 
        
        # FILTER OUT THE CLASSIFIER
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
        
        # Load the filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        return model, checkpoint.get('label_encoder_classes')

def train_one_epoch(loader, model, arcface_loss, scaler, optimizer, margin=1.0):
    model.train()
    total = 0.0
    total_triplet = 0.0
    total_ce = 0.0
    total_compact = 0.0
    total_species = 0.0
    criterion_ce = nn.CrossEntropyLoss(label_smoothing =0.001)

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none = True)

        # reshape(-1, 2) ensures it splits the pairs correctly, then we separate them
        y_stacked = data.y.view(-1, 2).to(device) 
        labels = y_stacked[:, 0]          # Identity labels
        species_labels = y_stacked[:, 1]  # Species labels
        with torch.amp.autocast():
            # Unpack the two outputs
            emb, species_logits, features = model(data)

            # Calculate losses
            #loss_triplet = batch_hard_triplet_loss(emb, labels, margin=margin)
            loss_triplet = batch_topk_semi_hard_triplet_loss(features, labels, margin=margin, k_neg=8)
            #TODO add args to change betwen arcface and cross entorpy
            #loss_ce = 0.25 * criterion_ce(logits, labels)
            loss_arc = 0.2 * arcface_loss(emb,labels)
            #loss_compact = 3 * batch_compactness_loss(features, labels)
            loss_ce_species = 1 * criterion_ce(species_logits, species_labels)
            # Combined Loss
            loss = loss_triplet + loss_arc + loss_ce_species # loss_compact

        #optimizer.zero_grad(set_to_none=True)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        total_triplet += loss_triplet.item()
        total_ce += loss_arc.item()
        #total_compact += loss_compact
        total_species += loss_ce_species.item()

    return total / len(loader), total_triplet / len(loader), total_ce / len(loader), total_compact / len(loader), total_species / len(loader)


def train(loader, model, optimizer, num_epochs, arcface_loss, start_epoch=0, args=None, train_classes = None, species_classes = None, scheduler = None):
    scaler = torch.amp.GradScaler()
    
    for i in range(start_epoch, num_epochs):

        total_batchloss, triplet, ce, compact, species = train_one_epoch(loader, model, arcface_loss, scaler, optimizer, margin=1)
        print(f"epoch {i} batchloss: {total_batchloss}, triplet loss: {triplet}, Arcface loss: {ce}, compactness loss: {compact}, ce species loss {species}")
        if scheduler is not None:
            scheduler.step(total_batchloss)
        #if i % 2 == 0:
            #torch.cuda.empty_cache()
        if i % 2 ==0:
            _ = model.save(args, epoch=i, optimizer=optimizer, train_classes=train_classes, species_classes=species_classes)


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
    csv_path = args.csv_path
    img_root = args.root_dir
    cache_pool = args.cache_dir
    #for resume only
    print(f"\n[ Setting up Output Directory: {args.checkpoint_dir} ]")
    
    if os.path.exists(args.checkpoint_dir):
        if args.resume:
            # logic constraint met: directory exists AND resume is set. Safe to continue.
            print(f"--> Directory exists. Resuming training.")
        else:
            # logic constraint violated: directory exists but we aren't resuming.
            # Abort to prevent accidental overwriting of a previous run's results.
            print(f"\n[ERROR] The checkpoint directory '{args.checkpoint_dir}' already exists.")
            print("To prevent accidental overwriting of previous results, this script is aborting.")
            print("\nTo fix this:")
            print("1. If you want to RESUME, add '--resume path/to/previous/checkpoint.pth'")
            print("2. If you want a FRESH run, change '--checkpoint_dir' to a new name in your command.")
            return # Exit the main function 
    else:
        # Directory doesn't exist, this is a normal fresh run. Create it.
        print(f"--> Creating new directory.")
        # Use makedirs just in case parent directories are needed, exist_ok handled by logic above
        os.makedirs(args.checkpoint_dir, exist_ok=True) 
    # ------------------------------------------------------------
    # 1. Load the Universal Metadata
    print("\n[ Loading Metadata ]")
    df = pd.read_csv(csv_path)
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
        train_labels, test_labels = train_test_split(unique_labels, test_size=0.2, random_state=42)
        train_df = df[df["global_label"].isin(train_labels)].reset_index(drop=True)
        test_df = df[df["global_label"].isin(test_labels)].reset_index(drop=True)
        test_df.to_csv(f"{args.checkpoint_dir}/current_test_split.csv", index=False)

    
    # No matter how the split was made, we must ensure training labels 
    # are strictly 0 to (N-1) for the Cross Entropy classifier.
    print("\n[ Processing Labels ]")
    train_le = LabelEncoder()
    species_le = LabelEncoder()
    df['species_label'] = species_le.fit_transform(df['species'])
    train_df['contiguous_label'] = train_le.fit_transform(train_df['global_label'])  
    train_df['species_label'] = species_le.transform(train_df['species'])
    test_df['species_label'] = species_le.transform(test_df['species'])

    # Calculate the number of classes for the model init
    num_train_classes = len(train_le.classes_)
    num_species = len(species_le.classes_)
    print(f"--> Active Training Classes: {num_train_classes}")
    print(f"--> Active Species Classes: {num_species}")

    # create augumentations
    aug_csv_path = prepare_augmented_training_data(train_df=train_df, root_dir=args.root_dir, lmdb_dir=args.cache_dir,args=args)
    aug_train_df = pd.read_csv(aug_csv_path)

    # 4. Create Sample Lists (mapping path -> label)
    # Train uses the NEW contiguous labels
    train_samples = list(zip(aug_train_df["path"], aug_train_df["contiguous_label"], aug_train_df["species_label"]))
    
    # Test uses the global_labels (Evaluation and Triplet loss don't care about gaps)
    test_samples = list(zip(test_df["path"], test_df["global_label"], test_df["species_label"]))

    print(f"Train size: {len(train_samples)} images | Test size: {len(test_samples)} images")

    if 'latent' in args.features:
        # Create a highly specific filename based on the current run's parameters
        cae_filename = f"texture_encoder_seg{args.segments}_dim{args.cae_latent_dim}_size{args.img_size}.pth"
        cae_path = os.path.join(args.checkpoint_dir, cae_filename)
        
        # Save the path to args so the DataLoader can access it later!
        args.cae_path = cae_path 
        
        # If it doesn't exist, or the user forces a rebuild, train it now!
        if not os.path.exists(cae_path) or args.rebuild:
            print(f"\n[ TRIGGER: Missing Texture Encoder ({cae_filename}). Initiating Training Sequence ]")
            train_and_save_cae(train_df, args, cae_path, device)
        else:
            print(f"\n[ Found pre-trained Texture Encoder at {cae_path} ]")

    # --- Initialize Universal Datasets ---
    print("\n[ Preparing Training Data ]")
    train_dataset = UniversalGraphDataset(
        num_train_classes = num_train_classes,
        n_hops= args.n_hops,
        samples=train_samples, 
        root_dir=img_root, 
        cache_dir=cache_pool, 
        mode=args.data_mode,
        img_size= args.img_size,
        rebuild_cache=args.rebuild,
        features=args.features,
        cae_version=args.cae_version,
        cae_weights_path=args.cae_weights_path,
        cae_latent_dim=args.cae_latent_dim,
        felz_sigma= args.felz_sigma,
        felz_scale= args.felz_scale,
        num_bins= args.num_hog_bins,
        min_size= args.felz_min_size
            )

    # Look at the very first graph in the dataset to see how wide the features are
    first_graph = train_dataset[0]
    dynamic_in_dim = first_graph.x.shape[1]
    print(f"--> Dynamically detected Node Feature Dimension (in_dim): {dynamic_in_dim}")
    print("\n[ Preparing Test/Holdout Data ]")
    """
    test_dataset = UniversalGraphDataset(
        num_train_classes = num_train_classes,
        n_hops= args.n_hops,
        samples=test_samples, 
        root_dir=img_root, 
        cache_dir=cache_pool, 
        mode=args.data_mode,
        img_size= args.img_size,
        rebuild_cache=args.rebuild,
        features=args.features,
        cae_version=args.cae_version,
        cae_weights_path=args.cae_weights_path,
        cae_latent_dim=args.cae_latent_dim,
        felz_sigma= args.felz_sigma,
        felz_scale= args.felz_scale,
        num_bins= args.num_hog_bins,
        min_size= args.felz_min_size
        )"""
    # DataLoaders
    batch_sampler = PKBatchSampler(aug_train_df["global_label"].values, P=14, K=8)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=args.workers, persistent_workers=False, prefetch_factor=None)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.workers)

    # Model & Optimizer
    # Look at the first graph to find out the feature width dynamically
    first_graph = train_dataset[0]
    dynamic_in_dim = first_graph.x.shape[1]
    print(f"--> Dynamically detected Node Feature Dimension (in_dim): {dynamic_in_dim}")

    # Model & Optimizer (Notice in_dim is now dynamic)
    model = ReIDModel(num_classes=num_train_classes, num_species= num_species,in_dim=dynamic_in_dim, hidden_dim=512, gnn_out_dim=256, emb_dim=512, 
                      use_hybrid_pooling = False, edge_strategy=args.edge_strategy,k_neighbors=args.k_neighbors, features=args.features, cae_latent_dim=args.cae_latent_dim, num_hog_bins= args.num_hog_bins).to(device)
    
    arcface = ArcFaceLoss(num_classes=num_train_classes, embedding_size=512, margin=12, scale= 64).to(device)
    #optimizer = torch.optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=0.0001)
    optimizer = bnb.optim.Adam8bit(list(model.parameters()) + list(arcface.parameters()), lr=0.0001)
    #optimizer = AdaBelief(list(model.parameters()) + list(arcface.parameters()), lr=1e-3)
    #TODO try reduce on Plateau

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',        # We want the loss to minimize
        factor=0.5,        # Multiply current LR by 0.5 when stuck
        patience=10,        # Wait 10 epochs of no improvement
        threshold=0.005,    # The loss must improve by at least this much to reset the patience
        cooldown= 5,
        min_lr= 0.000001
    )
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
    train(train_loader, model, optimizer,arcface_loss=arcface, num_epochs=args.epochs, start_epoch=start_epoch, args=args, train_classes=train_labels.tolist(), \
           species_classes=species_le.classes_.tolist(), scheduler=scheduler)
    # --- Saving the Results ---
    print("\n[ Saving Model ]")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

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
    #parser.add_argument("--segments", type=int, default=300, help="Number of superpixels (SLIC segments)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of graph cache (ignore existing .pt files)")
    parser.add_argument("--img_size", type=int, default=1024, help="Max dimension (width or height) for images before graph creation")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of identities to keep (e.g., 0.1 for 10%)")
    parser.add_argument("--species", type=str, nargs="+", default=None, help="List of specific species to use (e.g., --species tiger fox wolf)")

    
    parser.add_argument("--n_hops", type=int, default=1, help="Number of hops for edge connections (1 = direct neighbors, 2 = neighbors of neighbors)")
    
    # Graph Structure Settings
    parser.add_argument("--edge_strategy", type=str, default="spatial", choices=["spatial", "attention", "hybrid"], help="How to build GNN edges: 'spatial' (LMDB), 'attention' (GPU KNN), or 'hybrid' (Both).")
    parser.add_argument("--k_neighbors", type=int, default=3, help="Number of dynamic attention edges per node (if using attention or hybrid).")
    parser.add_argument("--felz_sigma", type=float, default=0.65, help = "")
    parser.add_argument("--felz_scale", type= float, default = 70, help = "")
    parser.add_argument("--felz_min_size", type= int, default = 150, help = "")
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
    parser.add_argument("--features", nargs="+", default=["color", "pos", "hog", "lbp", "cae", "texture"], help="List of node features to extract (color pos hog lbp texture)")
    parser.add_argument("--num_hog_bins", type=int, help="Historgram of oriented gradients bin size")
    # CAE settings
    parser.add_argument("--cae_latent_dim", type=int, default=64, help="Dimensionality of the CAE learned texture vector")
    parser.add_argument("--cae_epochs", type=int, default=10, help="Epochs to train the Texture Encoder")
    parser.add_argument("--cae_margin", type=float, default=0.5, help="Triplet loss margin for the Texture Encoder")
    parser.add_argument("--cae_version", type=str, default="v1_dim128", help="Version string of the CAE model (used for LMDB cache versioning)")
    parser.add_argument("--cae_weights_path", type=str, default="models/cae/v1_dim128/weights.pth", help="Path to the trained CAE weights")


    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k", help="Base directory for images")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--cache_dir", type=str, default="src/images/reid-10k/graph_cache_pool", help="Cache directory for graphs")

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
#TODO Triplet tracker evaluator