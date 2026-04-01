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
from joblib import Parallel, delayed

def process_single_image(filename, root_dir, cache_dir, n_segments, rebuild):
    """
    Worker function: Processes one image and saves its graph to disk.
    Returns True if processed, False if skipped or failed.
    """
    # 1. Create a safe, hierarchical cache path
    # We use the dataset folder name to keep the cache organized
    parts = filename.split('/')
    dataset_name = parts[1] if len(parts) > 1 else "unknown"
    safe_filename = os.path.basename(filename).rsplit('.', 1)[0] + ".pt"
    
    cache_subdir = os.path.join(cache_dir, dataset_name)
    cache_path = os.path.join(cache_subdir, f"seg{n_segments}_{safe_filename}")

    # 2. Skip if already exists (unless rebuilding)
    if not rebuild and os.path.exists(cache_path):
        return False

    # 3. Process
    try:
        os.makedirs(cache_subdir, exist_ok=True)
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert("RGB")
        
        # Generate the graph using your GNN utility
        graph = image_to_superpixel_graph(img, n_segments=n_segments)
        
        # Save to disk
        torch.save(graph, cache_path)
        return True
    except Exception as e:
        # We don't want one bad image to crash the whole 140k run
        return f"Error {filename}: {str(e)}"
    
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
    
class UniversalGraphDataset(PyGDataset):
    def __init__(self, samples, root_dir, cache_dir, mode='auto', n_segments=300, rebuild_cache=False):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.n_segments = n_segments
        self.graphs = []
        
        # 1. Determine Mode
        if mode == 'auto':
            # Threshold: ~8000 graphs is roughly 3-4 GB of RAM. 
            self.mode = 'memory' if len(samples) <= 8000 else 'lazy'
        else:
            self.mode = mode.lower()

        print(f"--> Initializing Dataset in [{self.mode.upper()}] mode for {len(samples)} samples.")

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # 2. Pre-computation / Warmup Phase
        # Even in lazy mode, we want to ensure all graphs exist on disk before training starts
        # so we don't bottleneck the GPU during epoch 1.
        self._warmup_cache(rebuild_cache)

        # 3. Memory Loading (Only if in memory mode)
        if self.mode == 'memory':
            print("--> Loading graphs into RAM...")
            for filename, label in tqdm(self.samples, desc="RAM Loading"):
                cache_path = self._get_cache_path(filename)
                graph = torch.load(cache_path, weights_only=False)
                graph.y = torch.tensor([int(label)], dtype=torch.long)
                self.graphs.append(graph)

    def _get_cache_path(self, filename):
        # Replaces slashes with underscores to flatten directory structure in cache safely
        safe_filename = filename.replace('/', '_').rsplit('.', 1)[0]
        graph_id = f"seg{self.n_segments}_{safe_filename}.pt"
        return os.path.join(self.cache_dir, graph_id)

    def _warmup_cache(self, rebuild):
        print(f"\n[ CACHE WARMUP ] Checking {len(self.samples)} samples...")
        
        # We pass n_jobs=-1 to use ALL available CPU cores.
        # If you want to leave some cores for browsing/other tasks, use -2 or -4.
        n_jobs = -2
        
        # The 'delayed' wrapper prepares the function calls
        tasks = (
            delayed(process_single_image)(
                filename, 
                self.root_dir, 
                self.cache_dir, 
                self.n_segments, 
                rebuild
            ) 
            for filename, _ in self.samples
        )

        print(f"--> Launching Parallel Warmup using {os.cpu_count()} cores...")
        
        # Run the tasks and wrap with tqdm for the status bar
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            tqdm(tasks, total=len(self.samples), desc="Generating Graphs", unit="img")
        )

        # Count successes/errors
        processed = sum(1 for r in results if r is True)
        errors = [r for r in results if isinstance(r, str)]
        
        print(f"\n[ WARMUP COMPLETE ]")
        print(f"--> New graphs created: {processed}")
        print(f"--> Images skipped (already cached): {len(results) - processed - len(errors)}")
        if errors:
            print(f"--> Errors encountered: {len(errors)} (check logs)")

    def len(self):
        return len(self.samples)

    def get(self, idx):
        if self.mode == 'memory':
            # O(1) RAM access
            return self.graphs[idx]
        else:
            # Disk streaming
            filename, label = self.samples[idx]
            cache_path = self._get_cache_path(filename)
            graph = torch.load(cache_path, weights_only=False)
            graph.y = torch.tensor([int(label)], dtype=torch.long)
            return graph