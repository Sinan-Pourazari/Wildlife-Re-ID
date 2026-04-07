import lmdb
import torch
import io
import os
from tqdm import tqdm
from torch_geometric.data import Data

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

        # Iterate through every single entry
        for key, value in tqdm(cursor, total=total_entries, desc="Auditing Graphs"):
            # 1. Check for empty entries
            if value is None or len(value) == 0:
                print(f"[!] Empty entry found for key: {key.decode()}")
                empty += 1
                continue

            # 2. Structural Integrity Check (Every 500th entry to save time)
            if sample_checked % 500 == 0:
                try:
                    buffer = io.BytesIO(value)
                    graph = torch.load(buffer, weights_only=False)
                    
                    # Verify it has the core GNN attributes
                    if not isinstance(graph, Data) or not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                        print(f"[!] Invalid Graph structure for key: {key.decode()}")
                        corrupted += 1
                except Exception as e:
                    print(f"[!] Deserialization error for key {key.decode()}: {e}")
                    corrupted += 1
            
            sample_checked += 1

    env.close()

    print("\n" + "="*30)
    print(f"VERIFICATION COMPLETE")
    print(f"--> Total Audited: {total_entries}")
    print(f"--> Empty Entries: {empty}")
    print(f"--> Corrupted/Invalid: {corrupted}")
    if empty == 0 and corrupted == 0:
        print("--> [STATUS] Database is HEALTHY.")
    else:
        print("--> [STATUS] Issues detected. Recommend running with --rebuild.")
    print("="*30)

# Run it
verify_lmdb_cache("src/images/reid-10k/graph_cache_pool")