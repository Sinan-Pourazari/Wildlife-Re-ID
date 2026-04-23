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
import gc
import torchvision.transforms as T
from rembg import new_session
import re
import torchvision.transforms.functional as TF
from PIL import Image


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
from rembg import remove, new_session

_WORKER_REMBG_SESSION = None


def get_rembg_session():
    """Loads the U2-Net session once per CPU worker to prevent RAM explosion."""
    global _WORKER_REMBG_SESSION
    if _WORKER_REMBG_SESSION is None:
        _WORKER_REMBG_SESSION = new_session("u2net")
    return _WORKER_REMBG_SESSION


def generate_augmented_metadata(train_df, target_K, save_dir):
    """
    PURE PLANNER: Calculates required augmentations to satisfy PKBatchSampler.
    Does NOT generate graphs. Only builds the Fat CSV for the Dataloader.
    """
    print(f"\n[ Pre-Flight ] Planning Augmentation Targets (Fill-to-K={target_K})...")
    id_counts = train_df['contiguous_label'].value_counts().to_dict()
    augmented_rows = []

    for index, row in train_df.iterrows():
        base_filename = row['path']
        label = row['contiguous_label']
        current_count = id_counts[label]

        # Calculate dynamic targets
        if current_count >= target_K:
            num_augs_to_make = 1
        else:
            num_augs_to_make = int(np.ceil((target_K - current_count) / current_count))

        # Generate virtual filenames
        for i in range(num_augs_to_make):
            # We inject the tag "_aug_" so the Dataset worker knows to apply augmentations
            aug_filename = base_filename.replace('.jpg', f'_aug_{i}.jpg')
            
            new_row = row.copy()
            new_row['path'] = aug_filename
            augmented_rows.append(new_row)

    # Compile the Fat CSV
    fat_train_df = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    fat_csv_path = os.path.join(save_dir, "train_metadata_augmented.csv")
    fat_train_df.to_csv(fat_csv_path, index=False)
    
    print(f"[ Pre-Flight ] Planned {len(augmented_rows)} new augmentations. Total graphs to verify: {len(fat_train_df)}.")
    return fat_csv_path


def process_for_lmdb(filename, root_dir, felz_scale, felz_sigma, min_size, num_bins, features, max_size, n_hops, cae_weights_path, cae_latent_dim):
    is_aug = '_aug_' in filename
    physical_filename = re.sub(r'_aug_\d+', '', filename) if is_aug else filename
    
    # root_dir now safely points to RAW images
    img_path = os.path.join(root_dir, physical_filename)
    img = Image.open(img_path).convert("RGB")

    # Pure image augmentations (no mask syncing needed anymore)
    if is_aug:
        import random
        if random.random() > 0.5:
            img = TF.hflip(img)
            
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        img = TF.resized_crop(img, i, j, h, w, size=[max_size, max_size], interpolation=Image.Resampling.LANCZOS)

        color_augmenter = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.3),
            T.RandomAutocontrast(p=0.2)
        ])
        img = color_augmenter(img)
    else:
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
    # Pass mask=None to the graph generator TODO remove deprecatd mask behaviour entirely
    graph = image_to_superpixel_graph(
        img, 
        mask=None, 
        scale=felz_scale,        
        sigma=felz_sigma,        
        min_size=min_size,       
        hog_bins=num_bins,      
        n_hops=n_hops, 
        features=features, 
        cae_weights_path=cae_weights_path,
        cae_latent_dim=cae_latent_dim
    )
    
    buffer = io.BytesIO()
    torch.save(graph, buffer)
        
    return True, buffer.getvalue()
    
def process_single_image(filename, root_dir, cache_dir, n_segments, rebuild, max_size=1024):
    # --- 1. Identify Dataset Folder ---
    parts = filename.split('/')
    dataset_name = parts[1] if len(parts) > 1 else "unknown"
    
    # Create the subfolder path
    cache_subdir = os.path.join(cache_dir, dataset_name)
    
    # Create filename
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
        # Force every single image to be exactly as defined in args
        # 1. Resize while keeping aspect ratio, and pad the rest with black
        img = ImageOps.pad(img, (max_size, max_size), color=(0, 0, 0), method=Image.Resampling.LANCZOS)

        from gnn.gnn import image_to_superpixel_graph
        graph = image_to_superpixel_graph(img, n_segments=n_segments)
        
        torch.save(graph, cache_path)
        return True
    except Exception as e:
        return f"Error {filename}: {str(e)}"

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
    def __init__(self, samples, root_dir, cache_dir, n_hops, felz_scale, felz_sigma, min_size, num_bins, mode='auto', rebuild_cache=False, img_size=1024,
                  num_train_classes=None, features=['color', 'pos', 'hog'], cae_version="none", cae_weights_path=None, cae_latent_dim = None):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.cache_dir = os.path.abspath(cache_dir)  # This is now the directory holding data.mdb and lock.mdb
        self.felz_scale = felz_scale
        self.felz_sigma = felz_sigma
        self.img_size = img_size
        self.graphs = []
        self.features = features
        self.n_hops = n_hops
        self.cae_version = cae_version
        self.cae_weights_path = cae_weights_path
        self.cae_latent_dim = cae_latent_dim
        self.min_size = min_size
        self.num_bins = num_bins
        self.base_key_bytes = self._generate_base_key()
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

        #TODO this seems like it wastes cpu cycles rework it!
        # strictly using env_key instead of self.cache_dir
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

    def _generate_base_key(self):
        """Calculates the static configuration string once."""
        feature_str = "-".join(sorted(self.features)) 
        
        if 'cae' in self.features:
            dim = getattr(self, 'cae_latent_dim')
            version = getattr(self, 'cae_version')
            feature_str += f"-CAE-v{version}-dim{dim}"
            
        if 'hog' in self.features:
            feature_str += f"hogb-{self.num_bins}"    
            
        base_str = f"res{self.img_size}_felzscale{self.felz_scale}_felzsigma{self.felz_sigma}_{self.min_size}_hops{self.n_hops}_{feature_str}_"
        return base_str.encode('utf-8')

    def _get_key(self, filename):
        """fast byte concatenation for rappid itteration."""
        return self.base_key_bytes + filename.encode('utf-8')
    def _warmup_cache(self, rebuild):

        print(f"\n[ CACHE WARMUP ] Checking {len(self.samples)} samples against LMDB...")
        # If texture is requested, bake the CAE version right into the key!
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
        # Inside _warmup_cache in dataloader.py
        def wrapper(task):
            import torch
            torch.set_num_threads(1)  # <--- STOPS CPU THRASHING
            key, filename = task
            
                    # Use explicit keywords to prevent positional mismatches!
            success, result = process_for_lmdb(
                filename=filename, 
                root_dir=self.root_dir, 
                felz_scale=self.felz_scale,
                felz_sigma=self.felz_sigma,
                min_size=self.min_size,
                num_bins=self.num_bins,
                features=self.features,
                max_size=self.img_size,
                n_hops=self.n_hops, 
                cae_weights_path=self.cae_weights_path,
                cae_latent_dim=self.cae_latent_dim # <--- ADD THIS
            )
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
        
        if idx % 500 == 0:  
            gc.collect()
        
        return graph
            
    def __del__(self):
        # We let the class-level manager handle LMDB cleanup safely.
        pass