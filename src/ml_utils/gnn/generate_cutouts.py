import os
import argparse
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import argparse
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm
import pandas as pd
import concurrent.futures

def process_single_image(filename, args, session):
    img_path = os.path.join(args.img_root, filename)
    cutout_path = os.path.join(args.cutout_root, filename)
    
    # THE FIX: Tell the thread to create the subfolders before saving!
    os.makedirs(os.path.dirname(cutout_path), exist_ok=True)
    
    try:
        img = Image.open(img_path).convert("RGB")
        mask = remove(img, session=session, only_mask=True).convert("L")
        black_bg = Image.new("RGB", img.size, (0, 0, 0))
        cutout = Image.composite(img, black_bg, mask)
        cutout.save(cutout_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")


def generate_offline_cutouts(args, df):
    os.makedirs(args.cutout_root, exist_ok=True)
    
    missing_files = [f for f in df['path'].tolist() if not os.path.exists(os.path.join(args.cutout_root, f))]
            
    if not missing_files:
        print(f"--> [CACHE HIT] All {len(df)} required cutouts already exist. Skipping U2-Net.")
        return

    print(f"--> [CACHE MISS] Generating {len(missing_files)} missing cutouts...")
    
    # Force ONNX to use your GPU
    session = new_session("u2net", providers=['CUDAExecutionProvider']) 
    
    # Use your 8 CPU cores to feed the GPU faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_image, filename, args, session) for filename in missing_files]
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(missing_files), desc="Cutting out backgrounds"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--img_root", type=str, default="src/images/reid-10k")
    parser.add_argument("--cutout_root", type=str, default="src/images/reid-10k-cutouts")
    
    # The filtering arguments!
    parser.add_argument("--species", type=str, nargs="+", default=None)
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print(f"Analyzing dataset targets...")
    df = pd.read_csv(args.csv_path) # It's already filtered by Step 0!
    generate_offline_cutouts(args, df)

def main(args):
    df = pd.read_csv(args.csv_path)
    
    # CRITICAL: Exclude Lynx from the U2-Net processing pipeline
    # (Adjust the string 'Lynx' to match exactly how the folder is named in your 'path' column)
    df_to_process = df[~df['path'].str.contains('Lynx', case=False, na=False)]
    
    print(f"Filtered out pre-segmented Lynx data. Processing {len(df_to_process)} remaining images")
    generate_offline_cutouts(args, df_to_process)