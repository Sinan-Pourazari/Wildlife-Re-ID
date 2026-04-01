import random
from torch.utils.data import Dataset
import cv2 as cv
import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset as PyGDataset
from gnn.gnn import image_to_superpixel_graph 


class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = [label for _, label in base_dataset.samples]

        # build label → indices mapping
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __getitem__(self, index):
        # anchor
        anchor_img, anchor_label = self.base_dataset[index]

        # positive (same class, not same index)
        pos_index = index
        while pos_index == index:
            pos_index = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self.base_dataset[pos_index]

        # negative (different class)
        neg_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        neg_index = random.choice(self.label_to_indices[neg_label])
        negative_img, _ = self.base_dataset[neg_index]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.base_dataset)



class TripletTrainDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        csv_path: path to training CSV with columns [animal_id, filename]
        img_dir: folder where training images are stored
        transform: torchvision transforms to apply to each image
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # group file indices by animal_id
        self.label_to_indices = {}
        for idx, row in self.df.iterrows():
            label = row["animal_id"]
            self.label_to_indices.setdefault(label, []).append(idx)

    def __getitem__(self, index):
        # anchor
        anchor_row = self.df.iloc[index]
        anchor_path = os.path.join(self.img_dir, anchor_row["filename"])
        anchor_img = cv.imread(anchor_path)
        #anchor_img = Image.open(anchor_path).convert("RGB")
        anchor_label = anchor_row["animal_id"]

        if self.transform:
            anchor_img = self.transform(anchor_img)
            print(type(anchor_img))
            print(anchor_img.shape)
            print(anchor_img)

        # positive (same label, different index)
        pos_index = index
        while pos_index == index:
            pos_index = random.choice(self.label_to_indices[anchor_label])
        pos_row = self.df.iloc[pos_index]
        pos_path = os.path.join(self.img_dir, pos_row["filename"])

        positive_img = cv.imread(pos_path)
        #positive_img = Image.open(pos_path).convert("RGB")
        if self.transform:
            positive_img = self.transform(positive_img)

        # negative (different label)
        neg_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        neg_index = random.choice(self.label_to_indices[neg_label])
        neg_row = self.df.iloc[neg_index]
        neg_path = os.path.join(self.img_dir, neg_row["filename"])
        negative_img = cv.imread(neg_path)
        #negative_img = Image.open(neg_path).convert("RGB")
        if self.transform:
            negative_img = self.transform(negative_img)
        #print(type(anchor_img))
        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.df)


class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        csv_path: path to test CSV with single column [filename]
        img_dir: folder where test images are stored
        transform: torchvision transforms to apply to each image
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        animal_id = row["animal_id"]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # animal_id = row number + 1 (1-based indexing)

        label = animal_id
        return img, label

    def __len__(self):
        return len(self.df)
    


class InMemoryGraphDataset(PyGDataset):
    def __init__(self, samples, root_dir, cache_dir, n_segments=64):
        super().__init__()
        self.samples = samples  # List of (filename, label) from your CSV split
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.n_segments = n_segments
        self.graphs = []

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # 1. Console Prompt (Only ask once per execution)
        # We check a global flag so we don't ask for both Train and Val datasets
        if not hasattr(InMemoryGraphDataset, "_user_choice"):
            existing_count = len([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
            if existing_count > 0:
                choice = input(f"\n[CACHE] Found {existing_count} graphs in '{cache_dir}'. Use cache? (y/n): ").strip().lower()
                InMemoryGraphDataset._user_choice = (choice == 'y')
            else:
                InMemoryGraphDataset._user_choice = False

        # 2. Loading / Generation Phase
        print(f"Dataset Warmup: Preparing {len(samples)} samples...")
        for filename, label in tqdm(samples):
            # Map filename to .pt
            graph_id = filename.rsplit('.', 1)[0] + '.pt'
            cache_path = os.path.join(self.cache_dir, graph_id)

            if InMemoryGraphDataset._user_choice and os.path.exists(cache_path):
                # LOAD EXISTING
                graph = torch.load(cache_path, weights_only=False)
            else:
                # GENERATE NEW
                img_path = os.path.join(self.root_dir, filename)
                img = Image.open(img_path).convert("RGB")
                graph = image_to_superpixel_graph(img, n_segments=self.n_segments)
                
                # Save to the global pool
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(graph, cache_path)

            # Important: Assign the label dynamically from the CSV
            # This ensures that even if you change label mappings between 
            # Open/Closed sets, the graph remains valid.
            graph.y = torch.tensor([int(label)], dtype=torch.long)
            self.graphs.append(graph)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]