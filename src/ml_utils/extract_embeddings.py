from concurrent.futures import ThreadPoolExecutor
import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
from gnn.gnn import GNNEncoder  
import gnn.gnn as gnn


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")
#DEVICE = "cpu"
BATCH_SIZE = 1          # graph building is per-image
EMB_PRECISION = 6
OUTPUT_CSV = "embeddings/amur_all_embeddings.csv"


# -----------------------------
# CORE EXTRACTION FUNCTION
# -----------------------------
@torch.no_grad()
@torch.no_grad()
def append_dataset(
    dataset,
    model,
    writer,
    split_name,
    header_written
):
    for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):

        sample = dataset[i]

        # handle datasets with or without paths
        if len(sample) == 3:
            img, label, path = sample
        else:
            img, label = sample
            path = ""

        # build graph
        graph = gnn.image_to_superpixel_graph(img, hog_bins=16, n_segments=300).to(DEVICE)

        # forward pass
        emb = model(graph).squeeze(0).cpu().numpy()

        # write header once
        if not header_written[0]:
            header = (
                ["image_path", "label", "split"] +
                [f"emb_{j}" for j in range(len(emb))]
            )
            writer.writerow(header)
            header_written[0] = True

        # write row
        row = (
            [path, label, split_name] +
            [round(float(x), EMB_PRECISION) for x in emb]
        )

        writer.writerow(row)

def extract_split(dataset, split_name, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header_written = [False]
        append_dataset(dataset, model, writer, split_name, header_written)
        
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    os.makedirs("embeddings", exist_ok=True)

    # ---- Load frozen encoder ----
    # TODO parameterise + 16 (hog dim)
    model = GNNEncoder(in_dim=5 + 16, hidden_dim=512, out_dim=256)
    model.eval().to(DEVICE)

    # ---- Datasets (NO transform!) ----
    """    train_ds = dataloader.TrainDataset(
        "src/images/Amur_Tigers/reid_list_train.csv",
        "src/images/Amur_Tigers/train",
        transform=None
    )"""

    test_ds = dataloader.TestDataset(
        "src/images/Amur_Tigers/reid_list_test.csv",
        "src/images/Amur_Tigers/test",
        transform=None
    )

    train_ds = dataloader.TestDataset(
        "src/images/Amur_Tigers/reid_list_train.csv",
        "src/images/Amur_Tigers/train",
        transform=None
    )

    # ---- Single CSV ----
    header_written = [False]

    with open("embeddings/amur_train_embeddings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header_written = [False]
        append_dataset(
            train_ds,
            model,
            writer,
            "train",
            header_written
        )

    with open("embeddings/amur_test_embeddings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header_written = [False]
        append_dataset(
            test_ds,
            model,
            writer,
            "test",
            header_written
        )


    print(f" All embeddings written to {OUTPUT_CSV}")
