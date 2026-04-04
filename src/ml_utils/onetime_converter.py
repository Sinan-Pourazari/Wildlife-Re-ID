import os
import pandas as pd
import lmdb
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = "src/images/reid-10k/metadata.csv"
OLD_CACHE_DIR = "src/images/reid-10k/graph_cache_pool_old" 
LMDB_DIR = "src/images/reid-10k/graph_cache_pool"          # Where the new LMDB will live
N_SEGMENTS = 300
# ---------------------

def get_old_pt_path(filename, old_cache_dir, n_segments):
    """Reconstructs the path to the old .pt file based on your original dataloader logic."""
    parts = filename.split('/')
    dataset_name = parts[1] if len(parts) > 1 else "unknown"
    safe_filename = os.path.basename(filename).rsplit('.', 1)[0] + ".pt"
    return os.path.join(old_cache_dir, dataset_name, f"seg{n_segments}_{safe_filename}")

def get_lmdb_key(filename, n_segments):
    """Generates the key format expected by the new LMDB dataloader."""
    return f"seg{n_segments}_{filename}".encode('utf-8')

def main():
    print("Loading metadata...")
    # Load the CSV to get the exact filenames your dataset uses
    df = pd.read_csv(CSV_PATH).dropna(subset=['identity'])
    
    # We only need the unique image paths to process
    filenames = df['path'].unique()
    print(f"Found {len(filenames)} unique images in metadata.")

    # Prepare the new LMDB directory
    os.makedirs(LMDB_DIR, exist_ok=True)
    
    # 100GB map size (This allocates virtual memory, not physical disk space. Perfectly safe.)
    map_size = 100 * 1024 * 1024 * 1024 
    env = lmdb.open(LMDB_DIR, map_size=map_size)

    missing_files = 0
    converted_files = 0

    print(f"Converting .pt files from '{OLD_CACHE_DIR}' to LMDB...")
    
    # Open a single write transaction for max speed
    with env.begin(write=True) as txn:
        for filename in tqdm(filenames, desc="Migrating", unit="graph"):
            old_pt_path = get_old_pt_path(filename, OLD_CACHE_DIR, N_SEGMENTS)
            lmdb_key = get_lmdb_key(filename, N_SEGMENTS)

            if not os.path.exists(old_pt_path):
                missing_files += 1
                continue

            # Read the .pt file directly as raw bytes (bypasses PyTorch overhead)
            with open(old_pt_path, "rb") as f:
                file_bytes = f.read()

            # Write directly to LMDB
            txn.put(lmdb_key, file_bytes)
            converted_files += 1

    env.close()

    print("\n=== Conversion Complete ===")
    print(f"Successfully migrated: {converted_files} graphs.")
    if missing_files > 0:
        print(f"Missing .pt files (skipped): {missing_files}")
    print(f"\nYou can now delete '{OLD_CACHE_DIR}' once you verify the new pipeline works.")

if __name__ == "__main__":
    main()