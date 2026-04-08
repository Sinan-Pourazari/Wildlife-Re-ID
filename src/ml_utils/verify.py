import lmdb
import torch
import io
import os
import collections
from tqdm import tqdm
from torch_geometric.data import Data
def peek_at_unknowns(cache_dir, num_samples=15):
    print(f"Peeking at 'unknown' keys in LMDB: {cache_dir}...\n")
    env = lmdb.open(cache_dir, readonly=True, lock=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        found = 0
        
        for key, _ in cursor:
            key_str = key.decode('utf-8')
            
            # If it doesn't match your current naming convention
            if not key_str.startswith('res'):
                print(f"Unknown Key Found: {key_str}")
                found += 1
                
            if found >= num_samples:
                break
                
    env.close()

def verify_lmdb_cache(cache_dir):
    print(f"--> Opening LMDB at {cache_dir} for audit...")
    
    if not os.path.exists(cache_dir):
        print("[ERROR] Cache directory does not exist.")
        return

    # Open in read-only mode
    env = lmdb.open(cache_dir, readonly=True, lock=False)
    
    with env.begin() as txn:
        stats = env.stat()
        total_entries = stats['entries']
        print(f"--> Total entries found: {total_entries}")

        cursor = txn.cursor()
        corrupted = 0
        empty = 0
        sample_checked = 0
        
        # Dictionary to track how many graphs belong to each configuration
        config_counts = collections.defaultdict(int)

        # Iterate through every single entry
        for key, value in tqdm(cursor, total=total_entries, desc="Auditing Graphs"):
            key_str = key.decode('utf-8')
            
            # --- EXTRACT CONFIGURATION ---
            # Keys look like: res1024_seg300_color-pos-hog_filename.jpg
            parts = key_str.split('_')
            
            # We assume the first 3 chunks (res, seg, features) define the configuration
            if len(parts) >= 3 and parts[0].startswith('res'):
                config_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            else:
                config_name = "unknown_config"
                
            config_counts[config_name] += 1
            # -----------------------------

            # 1. Check for empty entries
            if value is None or len(value) == 0:
                print(f"[!] Empty entry found for key: {key_str}")
                empty += 1
                continue

            # 2. Structural Integrity Check (Every 500th entry to save time)
            if sample_checked % 500 == 0:
                try:
                    buffer = io.BytesIO(value)
                    graph = torch.load(buffer, weights_only=False)
                    
                    # Verify it has the core GNN attributes
                    if not isinstance(graph, Data) or not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                        print(f"[!] Invalid Graph structure for key: {key_str}")
                        corrupted += 1
                except Exception as e:
                    print(f"[!] Deserialization error for key {key_str}: {e}")
                    corrupted += 1
            
            sample_checked += 1

    env.close()

    # --- PRINT THE SUMMARY ---
    print("\n" + "="*40)
    print("           VERIFICATION COMPLETE")
    print("="*40)
    print(f"--> Total Audited:       {total_entries}")
    print(f"--> Empty Entries:       {empty}")
    print(f"--> Corrupted/Invalid:   {corrupted}")
    print("\n[ CONFIGURATION BREAKDOWN ]")
    
    # Sort the dictionary so the output is neat and readable
    for config, count in sorted(config_counts.items()):
        print(f"  • {config:<30} : {count} graphs")
        
    print("-" * 40)
    
    if empty == 0 and corrupted == 0:
        print("--> [STATUS] Database is HEALTHY.")
    else:
        print("--> [STATUS] Issues detected. Recommend running with --rebuild.")
    print("="*40)

# Run it
if __name__ == "__main__":
    verify_lmdb_cache("src/images/reid-10k/graph_cache_pool")
    peek_at_unknowns("src/images/reid-10k/graph_cache_pool")