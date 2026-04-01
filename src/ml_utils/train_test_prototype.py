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
from loss_mining_tools import batch_hard_triplet_loss, PKBatchSampler, batch_semi_hard_triplet_loss
from PIL import Image
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from gnn.gnn import image_to_superpixel_graph, GNNEncoder
import torch.nn.functional as F
from dataloader import InMemoryGraphDataset
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")
class SimpleDataset(TorchDataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)

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
    def __init__(self, gnn_out_dim=256, emb_dim=124):
        super().__init__()
        self.encoder = GNNEncoder(in_dim=14, hidden_dim=256, out_dim=gnn_out_dim)
        self.head = nn.Sequential(
            nn.Linear(gnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, data):
        z = self.encoder(data)
        z = self.head(z)
        z = F.normalize(z, dim=1)
        return z
    

def train_one_epoch(loader, model, optimizer, margin=1.0):
    model.train()
    total = 0.0

    for data in loader:
        data = data.to(device)
        labels = data.y.view(-1).to(device)

        emb = model(data)
        loss = batch_semi_hard_triplet_loss(emb, labels, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item())

    return total / len(loader)


def train(loader, model, optimizer, num_epochs):
    for i in range(num_epochs):
        batchloss = train_one_epoch(loader, model, optimizer, margin=0.5)
        print(f"epoch {i} batchloss: {batchloss}")


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
    model.eval()

    emb_list = []
    label_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            labels = data.y.view(-1).to(device)

            emb = model(data)

            emb_list.append(emb)
            label_list.append(labels)

    embeddings = torch.cat(emb_list).cpu()
    labels = torch.cat(label_list).cpu()

    if closed_set:
        acc = knn_accuracy(
            embeddings.numpy(),
            labels.numpy()
        ) * 100

        COLOR = accuracy_to_color(acc)
        RESET = "\033[0m"

        print(
            f"Validation accuracy in Closed-set problem setting: "
            f"{COLOR}{acc:.2f}%{RESET}"
        )
        return

    memory = ec.IdentityMemory(
        threshold=0.10,
        max_exemplars_per_identity=5
    )

    predicted_ids = []

    for emb in embeddings:
        identity_id, _, _ = memory.upsert(emb)
        predicted_ids.append(identity_id)

    predicted_ids = torch.tensor(predicted_ids)

    ari = adjusted_rand_score(
        labels.numpy(),
        predicted_ids.numpy()
    )

    nmi = normalized_mutual_info_score(
        labels.numpy(),
        predicted_ids.numpy()
    )

    print("Open-set evaluation:")
    print(f"  Discovered identities : {len(memory.memory)}")
    print(f"  ARI (cluster quality) : {ari:.4f}")
    print(f"  NMI (label agreement): {nmi:.4f}")

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



def main(closed_set: bool):
    # --- Configuration & Paths ---
    train_csv = "src/images/Amur_Tigers/reid_list_train.csv"
    train_root = "src/images/Amur_Tigers/train"
    
    # We point both to a shared cache pool so the Open/Closed sets 
    # can reuse the same .pt files if they share images.
    global_cache = "src/images/Amur_Tigers/graph_cache_pool" 

    # --- Data Loading & Filtering ---
    df = pd.read_csv(train_csv)
    df = df.rename(columns={"animal_id": "label"})
    df["label"] = df["label"].astype(int)

    # Filter out identities with only one image (can't form triplets)
    counts = df["label"].value_counts()
    keep_ids = counts[counts > 1].index
    df = df[df["label"].isin(keep_ids)].reset_index(drop=True)

    print(f"After filtering: {len(df)} samples, {df['label'].nunique()} IDs")

    # --- Splitting Logic ---
    if closed_set:
        print("Mode: Closed Set (Random split, all identities seen in training)")
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    else:
        print("Mode: Open Set (ID split, validation identities are completely unseen)")
        unique_labels = df["label"].unique()
        train_labels, val_labels = train_test_split(unique_labels, test_size=0.3, random_state=42)
        train_df = df[df["label"].isin(train_labels)]
        val_df = df[df["label"].isin(val_labels)]

    # Prepare sample lists for the dataset
    train_samples = list(zip(train_df["filename"].tolist(), train_df["label"].tolist()))
    val_samples = list(zip(val_df["filename"].tolist(), val_df["label"].tolist()))

    # --- Warmup Phase (Cache/Memory Loading) ---
    # Using the same global_cache for both ensures we don't recompute 
    # graphs that appear in both sets or across different runs.
    print("\n--- Starting Train Dataset Warmup ---")
    train_dataset = InMemoryGraphDataset(train_samples, train_root, global_cache, n_segments=300)
    
    print("\n--- Starting Val Dataset Warmup ---")
    val_dataset = InMemoryGraphDataset(val_samples, train_root, global_cache, n_segments=300)

    # --- DataLoaders & Sampling ---
    y_train = train_df["label"].values
    P, K = 8, 4 # 8 Identities, 4 Images each = Batch size 32
    batch_sampler = PKBatchSampler(y_train, P=P, K=K)

    # We use the PyG DataLoader specifically designed for Graph Data objects
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"\nTrain dataset length: {len(train_dataset)}")
    print(f"Val dataset length:   {len(val_dataset)}")
    print(f"Train identities:    {len(set(train_df['label'].tolist()))}")
    print(f"Val identities:      {len(set(val_df['label'].tolist()))}")

    # --- Model Initialization ---
    # ReIDModel usually acts as a wrapper for your GNNEncoder 
    # to project the graph embedding into a contrastive space.
    model = ReIDModel(gnn_out_dim=256, emb_dim=124).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # --- Training & Evaluation ---
    print("\nStarting Training...")
    train(train_loader, model, optimizer, num_epochs=50)

    print("\nFinal Evaluation...")
    eval(model, val_loader, closed_set=closed_set)

if __name__ == "__main__":
    # Choose between Open Set or Closed Set testing
    main(closed_set=True)




#################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def reduce_to_2d(
    embeddings: torch.Tensor,
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
        return PCA(n_components=2, random_state=seed).fit_transform(X)

    if method.lower() == "tsne":
        # Standard trick: PCA -> t-SNE for speed/stability
        pca_dim = min(50, d, max(2, n - 1))
        Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

        # perplexity must be < n; choose a safe value
        # typical range: 5..30
        perplexity = min(30, max(5, (n - 1) // 3))
        perplexity = min(perplexity, n - 1)

        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
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
