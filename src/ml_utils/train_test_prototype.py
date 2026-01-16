import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import pandas as pd
import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import embedding_clusterings as ec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TripletDataset(Dataset):
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



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 126),
        )

    def forward(self, x):
        x = self.flatten(x)
        embeddings = self.linear_relu_stack(x)
        return embeddings
    

def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    batch_loss = 0.0
    for anchor, pos, neg in loader:
        anchor = anchor.to(device)
        pos    = pos.to(device)
        neg    = neg.to(device)
        #forwardpasses to generate embeddings for each triplet datapoint
        anchor_emb = model(anchor)
        pos_emb = model(pos)
        neg_emb = model(neg)

        #normalize embeddings
        anchor_emb = torch.nn.functional.normalize(anchor_emb, dim=1).to(device)
        pos_emb    = torch.nn.functional.normalize(pos_emb, dim=1).to(device)
        neg_emb    = torch.nn.functional.normalize(neg_emb, dim=1).to(device)

        #compute loss
        loss = loss_fn(anchor_emb,pos_emb,neg_emb)

        #backward pass
        optimizer.zero_grad()
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return batch_loss/len(loader)

def train(loader, model, optimizer, loss_fn, num_epochs):
    for i in range(num_epochs):
        batchloss = train_one_epoch(loader, model, optimizer, loss_fn)
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

    # --------------------------------------------------
    # 1. Extract embeddings (shared for both modes)
    # --------------------------------------------------
    with torch.no_grad():
        for features, label in loader:
            features = features.to(device)
            label = label.to(device)

            emb = torch.nn.functional.normalize(model(features), dim=1)

            emb_list.append(emb)
            label_list.append(label)

    embeddings = torch.cat(emb_list).cpu()   # (N, D)
    labels     = torch.cat(label_list).cpu() # (N,)

    # --------------------------------------------------
    # 2a. CLOSED-SET evaluation (classic Re-ID style)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2b. OPEN-SET evaluation (identity discovery)
    # --------------------------------------------------
    memory = ec.IdentityMemory(
        threshold=0.10,                 # tune later
        max_exemplars_per_identity=5
    )

    predicted_ids = []

    # IMPORTANT:
    # embeddings are processed SEQUENTIALLY
    for emb in embeddings:
        identity_id, _, _ = memory.upsert(emb)
        predicted_ids.append(identity_id)

    predicted_ids = torch.tensor(predicted_ids)

    # --------------------------------------------------
    # 3. Open-set evaluation metrics
    # --------------------------------------------------
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
    df = pd.read_csv("embeddings/amur_train_embeddings.csv")
    df.drop(columns=["image_path", "split"], inplace=True)

    # IMPORTANT: split features/labels by column name
    labels = df["label"].astype(int)
    features = df.drop(columns=["label"]).astype("float32")

    # Remove singleton IDs (must be done using labels)
    counts = labels.value_counts()
    keep_ids = counts[counts > 1].index
    mask = labels.isin(keep_ids)

    features = features[mask]
    labels   = labels[mask]

    print("After filtering:", len(labels), "samples,", labels.nunique(), "IDs")

    if not closed_set:
        animal_ids = labels.unique()
        train_ids, test_ids = train_test_split(animal_ids, test_size=0.3, random_state=42)

        train_mask = labels.isin(train_ids)
        test_mask  = labels.isin(test_ids)

        X_train, y_train = features[train_mask].values, labels[train_mask].values
        X_test,  y_test  = features[test_mask].values,  labels[test_mask].values
    else:
        X = features.values
        y = labels.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    test_dataset = SimpleDataset(X_test, y_test)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_dataset = TripletDataset(X_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length :", len(test_dataset))

    model = NeuralNetwork(input_dim=X_train.shape[1]).to(device)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-5)

    train(loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, num_epochs=50)
    print("Train identities:", len(set(y_train)))
    print("Test identities :", len(set(y_test)))
    eval(model, test_loader, closed_set=closed_set)


main(True)