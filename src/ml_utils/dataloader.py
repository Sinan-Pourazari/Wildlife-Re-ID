import random
import os
import io
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import lmdb
from torch_geometric.data import Dataset as PyGDataset
from torch.utils.data import Dataset
from gnn.gnn import image_to_superpixel_graph
from PIL import ImageOps
import numpy as np

def process_for_lmdb(filename, root_dir, n_segments, features, max_size, n_hops):
    img_path = os.path.join(root_dir, filename)
    img = Image.open(img_path).convert("RGB")

    # 1. Use the faster Thumbnail method (reduces total pixel area)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    # 2. Create the white mask to match the (now smaller) image
    mask = Image.new('L', img.size, color=255)

    from gnn.gnn import image_to_superpixel_graph
    # Pass the mask in to keep the graph safe from black edges
    graph = image_to_superpixel_graph(img, mask=np.array(mask), n_segments=n_segments, n_hops=n_hops, features=features)
        
    buffer = io.BytesIO()
    torch.save(graph, buffer)
        
    return True, buffer.getvalue()
    
def process_single_image(filename, root_dir, cache_dir, n_segments, rebuild, max_size=1024):
    # --- 1. Identify Dataset Folder ---
    # Assuming path is "images/DatasetName/..."
    parts = filename.split('/')
    dataset_name = parts[1] if len(parts) > 1 else "unknown"
    
    # Create the subfolder path
    cache_subdir = os.path.join(cache_dir, dataset_name)
    
    # Create filename: seg300_imagename.pt
    safe_filename = os.path.basename(filename).rsplit('.', 1)[0] + ".pt"
    cache_path = os.path.join(cache_subdir, f"seg{n_segments}_{safe_filename}")

    if not rebuild and os.path.exists(cache_path):
        return False

    try:
        # Ensure the dataset-specific subfolder exists
        os.makedirs(cache_subdir, exist_ok=True)
        
        img_path = os.path.join(root_dir, filename)
        img = Image.open(img_path).convert("RGB")

        # Resize Mechanic
        # Force every single image to be exactly 512x512
        # 1. Resize while keeping perfect aspect ratio, and pad the rest with black
        img = ImageOps.pad(img, (max_size, max_size), color=(0, 0, 0), method=Image.Resampling.LANCZOS)

        from gnn.gnn import image_to_superpixel_graph
        graph = image_to_superpixel_graph(img, n_segments=n_segments)
        
        torch.save(graph, cache_path)
        return True
    except Exception as e:
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
    _shared_envs = {}
    def __init__(self, samples, root_dir, cache_dir, n_hops, mode='auto', n_segments=300, rebuild_cache=False, img_size=1024, num_train_classes=None, features=['color', 'pos', 'hog']):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.cache_dir = os.path.abspath(cache_dir)  # This is now the directory holding data.mdb and lock.mdb
        self.n_segments = n_segments
        self.img_size = img_size
        self.graphs = []
        self.features = features
        self.n_hops = n_hops

        if mode == 'auto':
            self.mode = 'memory' if len(samples) <= 8000 else 'lazy'
        else:
            self.mode = mode.lower()

        print(f"--> Initializing Dataset in [{self.mode.upper()}] mode for {len(samples)} samples.")

        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Warmup / Generation Phase (CALLED ONLY ONCE NOW)
        self._warmup_cache(rebuild_cache)

        # 2. Memory Loading (Only if in memory mode)
        if self.mode == 'memory':
            self._load_all_to_memory()

    def _init_db(self, write=False):
        """Shared initialization to prevent 'Environment already open' errors."""
        pid = os.getpid()
        # Tie the environment handle to the specific Process ID!
        env_key = (self.cache_dir, pid)

        # FIXED: Now strictly using env_key instead of self.cache_dir
        if env_key not in UniversalGraphDataset._shared_envs:
            #print(f"{env_key=} does not exist!")
            inherited_keys = list(UniversalGraphDataset._shared_envs.keys())
            for idx, k in enumerate(inherited_keys):
                #print(f"Init db {idx=}, {k=}")
                if k[1] != pid:  # If the handle belongs to a different process (the parent)
                    #print(f"Closing {k=}")
                    try:
                        UniversalGraphDataset._shared_envs[k].close()
                    except Exception:
                        pass
                    del UniversalGraphDataset._shared_envs[k]

            # 100GB map size (virtual)
            map_size = 100 * 1024 * 1024 * 1024 
            
            UniversalGraphDataset._shared_envs[env_key] = lmdb.open(
                self.cache_dir,
                map_size=map_size,
                readonly=not write, # Warmup needs write=True, Workers need write=False
                lock=write,         # Only lock if we are writing
                readahead=False,
                meminit=False,
                max_readers=2048
            )
        else:
            #print(f"{env_key=} does not exist!")
            pass
        return UniversalGraphDataset._shared_envs[env_key]

    def _get_key(self, filename):
        """Standardized byte-key generator for LMDB, now feature-aware."""
        feature_str = "-".join(sorted(self.features)) 
        # FIX: Added img_size so 512 and 1024 are treated as completely different files!
        return f"res{self.img_size}_seg{self.n_segments}_hops{self.n_hops}_{feature_str}_{filename}".encode('utf-8')

    def _warmup_cache(self, rebuild):
        print(f"\n[ CACHE WARMUP ] Checking {len(self.samples)} samples against LMDB...")
        
        # 100GB map size. (This is virtual memory, it won't actually consume 100GB of disk space)
        env = self._init_db(write=True)

        # Gather existing keys to avoid redundant work
        with env.begin() as txn:
            existing_keys = set(txn.cursor().iternext(values=False))

        # Filter down to tasks that actually need processing
        tasks = []
        for filename, _, _ in self.samples:
            key = self._get_key(filename)
            if not rebuild and key in existing_keys:
                continue
            tasks.append((key, filename))

        if not tasks:
            print("--> All graphs present in LMDB. Skipping generation.")
            return

        print(f"--> Launching Parallel Warmup for {len(tasks)} missing graphs...")
        
        # Helper to wrap the task for Joblib
        def wrapper(task):
            import torch
            torch.set_num_threads(1)  # <--- STOPS CPU THRASHING
            
            key, filename = task
            success, result = process_for_lmdb(filename, self.root_dir, self.n_segments, self.n_hops, self.features, self.img_size)
            return key, success, result
        # return_as="generator" yields results as soon as workers finish them
        results_gen = Parallel(
            n_jobs=-1, 
            return_as="generator",
            batch_size=20,  # Send 20 images at a time to each worker
            pre_dispatch="2*n_jobs" # Ensure we don't overwhelm RAM
        )(delayed(wrapper)(task) for task in tasks)

        newly_created = 0
        errors = []
        buffer_limit = 200 # Write in chunks of 200 graphs
        current_buffer = []

        for key, success, result in tqdm(results_gen, total=len(tasks), desc="Writing to LMDB", unit="img"):
            if success:
                current_buffer.append((key, result))
                newly_created += 1
                
                if len(current_buffer) >= buffer_limit:
                    with env.begin(write=True) as txn:
                        for k, r in current_buffer:
                            txn.put(k, r)
                    current_buffer = [] # Clear buffer
            else:
                errors.append(result)
        
        # Final flush for any leftovers
        if current_buffer:
            with env.begin(write=True) as txn:
                for k, r in current_buffer:
                    txn.put(k, r)

        print(f"\n[ WARMUP COMPLETE ]")
        print(f"--> New graphs added to LMDB: {newly_created}")
        if errors:
            print(f"--> Errors encountered: {len(errors)}")

    def _load_all_to_memory(self):
        """Sequentially load all graphs into RAM (LMDB sequential read is extremely fast)."""
        print("--> Loading graphs into RAM from LMDB...")
        
        # FIXED: Capture the shared handle
        env = self._init_db(write=False)
        
        with env.begin() as txn:
            for filename, label, species_label in tqdm(self.samples, desc="RAM Loading"):
                key = self._get_key(filename)
                graph_bytes = txn.get(key)
                
                if graph_bytes is None:
                    raise KeyError(f"Graph not found in LMDB for key: {key.decode('utf-8')}")
                
                buffer = io.BytesIO(graph_bytes)
                graph = torch.load(buffer, weights_only=False)
                graph.y = torch.tensor([int(label), int(species_label)], dtype=torch.long)
                self.graphs.append(graph)

    def len(self):
        return len(self.samples)

    def get(self, idx):
        if self.mode == 'memory':
            return self.graphs[idx]
        else:
            # Get the shared handle
            env = self._init_db(write=False)
            
            filename, label, species_label = self.samples[idx]
            key = self._get_key(filename)
            
            with env.begin(write=False) as txn:
                graph_bytes = txn.get(key)
                
            if graph_bytes is None:
                raise KeyError(f"Missing key in LMDB: {key.decode('utf-8')}")
                
            buffer = io.BytesIO(graph_bytes)
            graph = torch.load(buffer, weights_only=False)
            graph.y = torch.tensor([int(label), int(species_label)], dtype=torch.long)
            
            return graph
            
    def __del__(self):
        # We let the class-level manager handle LMDB cleanup safely.
        pass