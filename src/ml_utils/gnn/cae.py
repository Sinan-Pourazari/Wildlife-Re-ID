import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from skimage.segmentation import slic
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageOps # Make sure ImageOps is imported!
# Import your existing tools!
import sys
from joblib import Parallel, delayed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from loss_mining_tools import batch_topk_triplet_loss, PKBatchSampler

import os
import cv2
import numpy as np
import torch
from skimage.segmentation import slic
from PIL import Image, ImageOps

def extract_patches_from_image(row_tuple, root_dir, img_size, n_segments, patches_per_image):
    """
    Standalone worker function to extract patches from a single image.
    Must be at the top level of the file for Windows multiprocessing to work!
    """
    # iterrows() passes a tuple of (index, Series), we only want the Series
    _, row = row_tuple 
    
    local_patches = []
    local_labels = []
    
    img_path = os.path.join(root_dir, row['path'])
    try:
        # Load and pad image
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.pad(img, (img_size, img_size), color=(0, 0, 0), method=Image.Resampling.LANCZOS)
        img = np.array(img)

        # Run SLIC
        segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)
        unique_segs = np.unique(segments)

        # Randomly pick superpixels
        chosen_segs = np.random.choice(unique_segs, min(patches_per_image, len(unique_segs)), replace=False)

        for sp in chosen_segs:
            mask = (segments == sp)
            coords = np.column_stack(np.nonzero(mask))

            if len(coords) < 10: 
                continue # Skip tiny fragments
            
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)

            # Crop and mask
            crop_img = img[ymin:ymax+1, xmin:xmax+1]
            crop_mask = mask[ymin:ymax+1, xmin:xmax+1]
            masked_crop = crop_img * crop_mask[..., np.newaxis]
            
            # Resize to 32x32 for the CAE
            patch_32 = cv2.resize(masked_crop.astype(np.uint8), (32, 32), interpolation=cv2.INTER_AREA)

            # Convert to PyTorch tensor [C, H, W]
            patch_tensor = torch.from_numpy(patch_32).float().permute(2, 0, 1) / 255.0
            
            local_patches.append(patch_tensor)
            local_labels.append(row_tuple[0])
            
    except Exception as e:
        # If an image is corrupt, the worker just returns empty lists safely
        pass
        
    return local_patches, local_labels


class TextureEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # --- ENCODER ---
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_enc = nn.Linear(128 * 4 * 4, latent_dim)

        # --- DECODER ---
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4)
        # ConvTranspose2d is the opposite of Pooling; it doubles the resolution
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)

    def encoder(self, x):
        """Used later by the GNN to get the 64-dim texture vector."""
        x = self.pool(F.relu(self.enc_conv1(x)))
        x = self.pool(F.relu(self.enc_conv2(x)))
        x = self.pool(F.relu(self.enc_conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def forward(self, x):
        """Used during training to reconstruct the image."""
        # 1. Compress
        z = self.encoder(x)
        
        # 2. Decompress
        x_recon = F.relu(self.fc_dec(z))
        x_recon = x_recon.view(x_recon.size(0), 128, 4, 4) # Reshape back to 4x4 image
        x_recon = F.relu(self.dec_conv1(x_recon))
        x_recon = F.relu(self.dec_conv2(x_recon))
        
        # Sigmoid pushes final pixels to be between 0.0 and 1.0 (matching your input)
        x_recon = torch.sigmoid(self.dec_conv3(x_recon)) 
        return x_recon
    
class SuperpixelPatchDataset(TorchDataset):
    def __init__(self, args, df, root_dir, n_segments, patches_per_image, n_jobs=-1):
        """
        Extracts random superpixels using parallel CPU processing.
        """
        self.patches = []
        self.labels = []
        
        print("\n[CAE Phase] Extracting Texture Patches from Images (Parallelized)...")

        # Sample a subset to prevent taking hours
        sample_df = df.sample(min(2000, len(df)), random_state=42)

        # Launch Parallel Workers! (n_jobs=-1 uses all available CPU cores)
        # return_as="generator" allows tqdm to update the progress bar in real-time
        results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(extract_patches_from_image)(
                row_tuple, root_dir, args.img_size, n_segments, patches_per_image
            ) for row_tuple in sample_df.iterrows()
        )

        # Gather the results as they finish
        for worker_patches, worker_labels in tqdm(results_gen, total=len(sample_df), desc="Extracting"):
            if worker_patches:  # If the worker didn't fail
                self.patches.extend(worker_patches)
                self.labels.extend(worker_labels)
                
        print(f"--> Successfully extracted {len(self.patches)} patches.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]
    

def train_and_save_cae(df, args, save_path, device):
    """The training of the Texture Autoencoder using Reconstruction Loss."""
    
    dataset = SuperpixelPatchDataset(
        args, df, root_dir=args.root_dir, 
        n_segments=args.segments, 
        patches_per_image=5,
        n_jobs=-1
    )
    
    # Standard Dataloader! No more complex Triplet Samplers.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = TextureEncoder(latent_dim=args.cae_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # We use Mean Squared Error (MSE) to measure how perfectly it rebuilds the image
    criterion = nn.MSELoss()
    
    epochs = args.cae_epochs
    print(f"\n[CAE Phase] Training Autoencoder for {epochs} epochs (Latent Dim: {args.cae_latent_dim})...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for patches, _ in loader: # We don't even need the labels!
            patches = patches.to(device)
            
            # Forward pass: Try to reconstruct the patch
            reconstruction = model(patches)
            
            # Calculate how far off the pixels are
            loss = criterion(reconstruction, patches)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} - Reconstruction Loss (MSE): {total_loss/len(loader):.4f}")
        
    print(f"--> Saving trained autoencoder to {save_path}")
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Train Texture Autoencoder")
    
    # We use 'dest' so the terminal can use '--epochs', but the Python code sees 'args.cae_epochs'
    parser.add_argument("--epochs", dest="cae_epochs", type=int, default=15)
    parser.add_argument("--latent_dim", dest="cae_latent_dim", type=int, default=64)
    parser.add_argument("--margin", dest="cae_margin", type=float, default=0.5)
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="models/cae")
    parser.add_argument("--model_name", type=str, default="texture_v1")
    
    # Dataset pathing
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--img_size", type=int, default=512)

    args = parser.parse_args()

    # 1. Setup device and directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")

    print(f"=========================================")
    print(f" INITIALIZING CAE STANDALONE TRAINING")
    print(f" Target: {save_path}")
    print(f" Device: {device}")
    print(f"=========================================")

    # 2. Load the metadata
    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] Could not find metadata CSV at {args.csv_path}")
        sys.exit(1)

    # 3. Ensure the dataframe has the columns your SuperpixelDataset expects
    if 'path' not in df.columns and 'filename' in df.columns:
        df['path'] = df['filename']
        
    if 'contiguous_label' not in df.columns and 'animal_id' in df.columns:
        # Create a 0-indexed ID list for Triplet Loss math
        df['contiguous_label'] = pd.factorize(df['animal_id'])[0]

    # 4. Launch the training sequence!
    train_and_save_cae(df, args, save_path, device)
    
    print("\n[SUCCESS] CAE standalone training completed safely.")