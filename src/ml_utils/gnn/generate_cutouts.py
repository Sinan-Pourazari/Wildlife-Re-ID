import os
import argparse
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm
import pandas as pd
import numpy as np
def generate_offline_cutouts(args, df):
    os.makedirs(args.cutout_root, exist_ok=True)
    
    # Check which files from the filtered list actually need processing
    missing_files = []
    for filename in df['path'].tolist():
        cutout_path = os.path.join(args.cutout_root, filename)
        if not os.path.exists(cutout_path):
            missing_files.append(filename)
            
    if not missing_files:
        print(f"--> [CACHE HIT] All {len(df)} required cutouts already exist. Skipping U2-Net.")
        return

    print(f"--> [CACHE MISS] Generating {len(missing_files)} missing cutouts...")
    session = new_session("u2net") 
    
    for filename in tqdm(missing_files, desc="Cutting out backgrounds"):
        img_path = os.path.join(args.img_root, filename)
        cutout_path = os.path.join(args.cutout_root, filename)
        os.makedirs(os.path.dirname(cutout_path), exist_ok=True)
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # 1. Generate the mask
            mask = remove(img, session=session, only_mask=True).convert("L")
            
            # 2. Create a pure black canvas of the same size
            black_bg = Image.new("RGB", img.size, (0, 0, 0))
            
            # 3. Paste the animal onto the black canvas using the mask
            cutout = Image.composite(img, black_bg, mask)
            
            # Save it
            cutout.save(cutout_path)
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

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
    df = pd.read_csv(args.csv_path) # It's already filtered!
    generate_offline_cutouts(args, df)