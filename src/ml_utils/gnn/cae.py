import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageOps
import sys
from joblib import Parallel, delayed
import random
import torchvision.transforms as T
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def get_gpu_power_watts():
    """Queries nvidia-smi for current GPU power draw in Watts."""
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                stdout=subprocess.PIPE, text=True
            )
            return float(result.stdout.strip().split('\n')[0])
        except Exception:
            return 0.0
    return 0.0

# --- RECONSTRUCTION AUGMENTATION ---
class ReconstructionTransform:
    """Standardizes patches for MSE reconstruction."""
    def __init__(self):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            # Normalization matches the cae_transform exactly in gnn.py
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)

def extract_patches_from_image(row_tuple, root_dir, img_size, args, patches_per_image):
    """Worker function: Extracts raw crops. unsupervised version"""
    _, row = row_tuple 
    local_patches = []
    patch_size = 64 
    
    img_path = os.path.join(root_dir, row['path'])
    try:
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.pad(img, (img_size, img_size), color=(0, 0, 0))
        img_np = np.array(img)
        
        mask = img_np.sum(axis=-1) > 0
        valid_y, valid_x = np.where(mask)
        
        if len(valid_y) == 0:
            valid_y, valid_x = [img_size // 2], [img_size // 2]
            
        valid_coords = list(zip(valid_y, valid_x))
        random.shuffle(valid_coords)
        
        patches_found = 0
        for (center_y, center_x) in valid_coords:
            if patches_found >= patches_per_image:
                break
                
            half_p = patch_size // 2
            left = max(0, min(center_x - half_p, img_size - patch_size))
            top = max(0, min(center_y - half_p, img_size - patch_size))
            right = left + patch_size
            bottom = top + patch_size
            
            patch_mask = mask[top:bottom, left:right]
            if patch_mask.mean() < 0.70:
                continue
                
            patch = img.crop((left, top, right, bottom))
            local_patches.append(np.array(patch))
            patches_found += 1
            
        while patches_found < patches_per_image:
            patch = img.crop((img_size//2 - patch_size//2, img_size//2 - patch_size//2, 
                              img_size//2 + patch_size//2, img_size//2 + patch_size//2))
            local_patches.append(np.array(patch))
            patches_found += 1
            
        return local_patches
        
    except Exception as e:
        return []

# 2. CONVOLUTIONAL AUTOENCODER (CAE)
class TextureEncoder(nn.Module):
    def __init__(self, latent_dim=32): 
        super().__init__()
        # ================= ENCODER =================
        # 64x64 -> 32x32
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.enc_bn1 = nn.BatchNorm2d(32)
        # 32x32 -> 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.enc_bn2 = nn.BatchNorm2d(64)
        # 16x16 -> 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.enc_bn3 = nn.BatchNorm2d(128)
        # 8x8 -> 4x4
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.enc_bn4 = nn.BatchNorm2d(256)
        
        # Base Encoder (This is what the GNN uses)
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # ================= DECODER =================
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # 4x4 -> 8x8
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        # 8x8 -> 16x16
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        # 16x16 -> 32x32
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        # 32x32 -> 64x64
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def encoder(self, x):
        """ representation layer called by gnn.py."""
        x = F.gelu(self.enc_bn1(self.enc_conv1(x)))
        x = F.gelu(self.enc_bn2(self.enc_conv2(x)))
        x = F.gelu(self.enc_bn3(self.enc_conv3(x)))
        x = F.gelu(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return F.normalize(z, p=2, dim=1)

    def forward(self, x):
        """Training pass goes through encoder and decoder."""
        # Encode
        z = self.encoder(x)
        
        # Decode
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 4, 4)
        h = F.gelu(self.dec_bn1(self.dec_conv1(h)))
        h = F.gelu(self.dec_bn2(self.dec_conv2(h)))
        h = F.gelu(self.dec_bn3(self.dec_conv3(h)))
        
        reconstruction = self.dec_conv4(h) 
        
        return reconstruction, z


class SuperpixelPatchDataset(TorchDataset):
    def __init__(self, args, df, root_dir, patches_per_image, n_jobs=-1):
        self.patches = []
        self.transform = ReconstructionTransform()
        
        print("\n[CAE Phase] Extracting Texture Patches (Unsupervised/No Labels)...")
        sample_df = df.sample(min(12000, len(df)), random_state=42)

        results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(extract_patches_from_image)(
                row_tuple, root_dir, args.img_size, args, patches_per_image
            ) for row_tuple in sample_df.iterrows()
        )

        for worker_patches in tqdm(results_gen, total=len(sample_df), desc="Extracting"):
            if worker_patches:  
                self.patches.extend(worker_patches)
                
        print(f"--> Extracted {len(self.patches)} raw patches for autoencoder training.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Convert numpy back to PIL and apply single transform
        pil_img = Image.fromarray(self.patches[idx])
        return self.transform(pil_img)

def train_and_save_cae(df, args, save_path, device):
    dataset = SuperpixelPatchDataset(
        args, df, root_dir=args.root_dir, 
        patches_per_image=25, n_jobs=-1
    )
    
    import multiprocessing
    optimal_workers = max(1, min(14, multiprocessing.cpu_count() - 2))
    print(f"\n--> Spawning {optimal_workers} background CPU workers for data loading...")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 256, 
        shuffle=True, 
        num_workers=optimal_workers, 
        pin_memory=True,          
        persistent_workers=True,  
        drop_last=True
    )
    
    model = TextureEncoder(latent_dim=args.cae_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss() # Replacing NT-Xent with simple MSE
    
    metrics_csv_path = os.path.join(args.checkpoint_dir, "cae_training_metrics.csv")
    headers = ["epoch", "reconstruction_loss", "vram_mb", "gpu_util_percent", "power_watts"]
    with open(metrics_csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")

    epochs = args.cae_epochs
    print(f"\n[CAE Phase] Training Convolutional Autoencoder for {epochs} epochs...")
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0  
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for view in loader:
            view = view.to(device)
            
            # Pass patch through network
            reconstruction, _ = model(view)
            
            # Calculate MSE against the original view
            loss = criterion(reconstruction, view)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(loader)

        vram_mb, gpu_util, power_watts = 0.0, 0.0, 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            try:
                gpu_util = torch.cuda.utilization()
            except Exception:
                pass
            power_watts = get_gpu_power_watts()
            
        print(f"  Epoch {epoch+1}/{epochs} - Reconstruction Loss (MSE): {epoch_loss:.4f} | VRAM: {vram_mb:.0f}MB | GPU: {gpu_util}%")
        
        row_data = [epoch+1, epoch_loss, vram_mb, gpu_util, power_watts]
        with open(metrics_csv_path, 'a') as f:
            f.write(",".join(map(str, row_data)) + "\n")
        
        # Track loss and handle early stopping
        if epoch_loss < best_loss - 0.0005:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[CAE Phase] Early Stopping triggered!")
                break

    print(f"\n--> Saving best unsupervised weights to disk...")
    model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), save_path)
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df_plot = pd.read_csv(metrics_csv_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_plot['epoch'], df_plot['reconstruction_loss'], label='MSE Loss', color='blue', linewidth=2)
        plt.title("CAE Reconstruction Training", fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, "cae_loss_plot.png"), dpi=300)
        plt.close()
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df_plot['epoch'], df_plot['power_watts'], label='Power (Watts)', color='red', linewidth=2)
        ax1.set_ylabel("Power (Watts)", color='red', fontweight='bold')
        ax2 = ax1.twinx() 
        ax2.plot(df_plot['epoch'], df_plot['gpu_util_percent'], label='GPU Util (%)', color='green', alpha=0.4, linewidth=2)
        ax2.set_ylabel("Utilization (%)", color='green', fontweight='bold')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        plt.title("CAE Hardware Utilization & Power", fontweight='bold')
        fig.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, "cae_hardware_plot.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"Failed to generate CAE plots: {e}")

    return save_path

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import sys

    parser = argparse.ArgumentParser(description="Train Reconstruction Texture Encoder")
    
    parser.add_argument("--epochs", dest="cae_epochs", type=int, default=15)
    parser.add_argument("--latent_dim", dest="cae_latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--save_dir", type=str, default="models/cae")
    parser.add_argument("--checkpoint_dir", type=str, default=".", help="Where to save telemetry")
    parser.add_argument("--model_name", type=str, default="texture_reconstruction")
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--img_size", type=int, default=256)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")

    print(f"=========================================")
    print(f" STARTING RECONSTRUCTION CAE TRAINING")
    print(f" Target: {save_path}")
    print(f"=========================================")

    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
        train_and_save_cae(df, args, save_path, device)
    except FileNotFoundError:
        print(f"[ERROR] Could not find metadata CSV at {args.csv_path}")
        sys.exit(1)