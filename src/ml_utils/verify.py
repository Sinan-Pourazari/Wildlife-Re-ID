import lmdb
import torch
import io
import os
import collections
from tqdm import tqdm
from torch_geometric.data import Data
import re

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
        
        # Track configurations and node dimensions
        config_counts = collections.defaultdict(int)
        dim_counts = collections.defaultdict(int)

        # Iterate through every single entry
        for key, value in tqdm(cursor, total=total_entries, desc="Auditing Graphs"):
            key_str = key.decode('utf-8')
            
            # --- EXTRACT CONFIGURATION ---
            # Extract the core parameters from the new key structure
            # Example: res512_felzscale70.0_felzsigma0.65_300_hops1_...
            match = re.match(r'^(res\d+_felzscale[\d\.]+_felzsigma[\d\.]+_\d+_hops\d+)', key_str)
            if match:
                config_name = match.group(1)
            else:
                config_name = "unknown_legacy_config"
                
            config_counts[config_name] += 1
            # -----------------------------

            # 1. Check for empty entries
            if value is None or len(value) == 0:
                empty += 1
                continue

            # 2. Structural Integrity & Dimension Check (Every 500th entry to save time)
            if sample_checked % 500 == 0:
                try:
                    buffer = io.BytesIO(value)
                    graph = torch.load(buffer, weights_only=False)
                    
                    # Verify it has the core GNN attributes
                    if not isinstance(graph, Data) or not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                        corrupted += 1
                    else:
                        # TRACK THE DIMENSION OF THE FEATURES!
                        in_dim = graph.x.shape[1]
                        dim_counts[in_dim] += 1
                        
                except Exception as e:
                    corrupted += 1
            
            sample_checked += 1

    env.close()

    # --- PRINT THE SUMMARY ---
    print("\n" + "="*60)
    print("                    VERIFICATION COMPLETE")
    print("="*60)
    print(f"--> Total Audited:       {total_entries}")
    print(f"--> Empty Entries:       {empty}")
    print(f"--> Corrupted/Invalid:   {corrupted}")
    print("-" * 60)
    
    print("\n[ GRAPH NODE DIMENSIONS (Sampled) ]")
    for dim, count in sorted(dim_counts.items()):
        estimated_total = count * 500
        print(f"  • {dim} Dimensions : ~{estimated_total} graphs")
        
        # Friendly diagnosis of the dimension sizes
        if dim == 38:
            print("    [!] WARNING: These are stale graphs! They are missing the 32-dim CAE features.")
        elif dim == 70:
            print("    [+] SUCCESS: These are fully updated 70-dim graphs with CAE injected.")

    print("\n[ CONFIGURATION BREAKDOWN ]")
    for config, count in sorted(config_counts.items()):
        print(f"  • {config} : {count} graphs")
        
    print("-" * 60)
    
    if empty == 0 and corrupted == 0:
        print("--> [STATUS] Database integrity is HEALTHY.")
    else:
        print("--> [STATUS] Issues detected. Recommend running with --rebuild.")
    print("="*60)

# Run it
if __name__ == "__main__":
    verify_lmdb_cache("src/images/reid-10k/graph_cache_pool")