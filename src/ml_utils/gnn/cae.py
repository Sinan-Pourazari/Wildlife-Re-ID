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

# --- 1. SIMCLR DUAL-VIEW AUGMENTATION ---
class ContrastiveTransform:
    """Applies two different random augmentations to the same patch."""
    def __init__(self):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def extract_patches_from_image(row_tuple, root_dir, img_size, args, patches_per_image):
    """Worker function: Extracts raw crops. NO LABELS NEEDED!"""
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
            # Return raw numpy array to survive joblib serialization easily
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

# --- 2. CONTRASTIVE TEXTURE ENCODER ---
class TextureEncoder(nn.Module):
    def __init__(self, latent_dim=32): 
        super().__init__()
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
        
        # Projection Head (Used ONLY for Contrastive Training)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def encoder(self, x):
        """The pure representation layer."""
        x = F.gelu(self.enc_bn1(self.enc_conv1(x)))
        x = F.gelu(self.enc_bn2(self.enc_conv2(x)))
        x = F.gelu(self.enc_bn3(self.enc_conv3(x)))
        x = F.gelu(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return F.normalize(z, p=2, dim=1)

    def forward(self, x):
        """Training pass goes through encoder AND projector."""
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, p=2, dim=1)

# --- 3. NT-Xent LOSS (Normalized Temperature-scaled Cross Entropy) ---
def nt_xent_loss(z1, z2, temperature=0.1):
    """Calculates contrastive loss between two augmented views of patches."""
    batch_size = z1.size(0)
    # Concatenate all views: [z1_1, ..., z1_B, z2_1, ..., z2_B]
    z = torch.cat([z1, z2], dim=0) 
    
    # Cosine similarity matrix
    sim_matrix = torch.exp(torch.mm(z, z.t()) / temperature)
    
    # Remove self-similarity from the diagonal
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    
    # Calculate positives (the similarity between z1 and its corresponding z2)
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    
    loss = -torch.log(positives / sim_matrix.sum(dim=-1))
    return loss.mean()

class SuperpixelPatchDataset(TorchDataset):
    def __init__(self, args, df, root_dir, patches_per_image, n_jobs=-1):
        self.patches = []
        self.transform = ContrastiveTransform()
        
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
                
        print(f"--> Extracted {len(self.patches)} raw patches for contrastive learning.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Convert numpy back to PIL and apply dual-transform
        pil_img = Image.fromarray(self.patches[idx])
        view_1, view_2 = self.transform(pil_img)
        return view_1, view_2

def train_and_save_cae(df, args, save_path, device):
    dataset = SuperpixelPatchDataset(
        args, df, root_dir=args.root_dir, 
        patches_per_image=5, n_jobs=-1
    )
    
    import multiprocessing
    # Use up to 8 CPU cores, but leave 2 free for the OS and GPU driver
    optimal_workers = max(1, min(14, multiprocessing.cpu_count() - 2))
    print(f"\n--> Spawning {optimal_workers} background CPU workers for on-the-fly augmentation...")
    
    # Standard random batching with working-ahead capabilities
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 256, 
        shuffle=True, 
        num_workers=optimal_workers, 
        pin_memory=True,          # Pre-allocates page-locked memory for instant GPU transfer
        persistent_workers=True,  # Keeps the CPU workers alive between epochs so they don't restart
        drop_last=True
    )
    
    model = TextureEncoder(latent_dim=args.cae_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    metrics_csv_path = os.path.join(args.checkpoint_dir, "cae_training_metrics.csv")
    headers = ["epoch", "contrastive_loss", "vram_mb", "gpu_util_percent", "power_watts"]
    with open(metrics_csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")

    epochs = args.cae_epochs
    print(f"\n[CAE Phase] Training Contrastive Texture Extractor for {epochs} epochs...")
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0  
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for view_1, view_2 in loader:
            view_1, view_2 = view_1.to(device), view_2.to(device)
            
            # Pass both views through encoder + projector
            z1 = model(view_1)
            z2 = model(view_2)
            
            loss = nt_xent_loss(z1, z2, temperature=0.1)
            
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
            
        print(f"  Epoch {epoch+1}/{epochs} - Contrastive Loss: {epoch_loss:.4f} | VRAM: {vram_mb:.0f}MB | GPU: {gpu_util}%")
        
        row_data = [epoch+1, epoch_loss, vram_mb, gpu_util, power_watts]
        with open(metrics_csv_path, 'a') as f:
            f.write(",".join(map(str, row_data)) + "\n")
        
        if epoch_loss < best_loss - 0.005:
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
    
    # Strip the projector before saving so it maps strictly to the base encoder
    del model.projector 
    torch.save(model.state_dict(), save_path)
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        df_plot = pd.read_csv(metrics_csv_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_plot['epoch'], df_plot['contrastive_loss'], label='NT-Xent Loss', color='blue', linewidth=2)
        plt.title("CAE Contrastive Training", fontweight='bold')
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

    parser = argparse.ArgumentParser(description="Train Contrastive Texture Encoder")
    
    parser.add_argument("--epochs", dest="cae_epochs", type=int, default=15)
    parser.add_argument("--latent_dim", dest="cae_latent_dim", type=int, default=32)
    parser.add_argument("--margin", dest="cae_margin", type=float, default=0.5) 
    parser.add_argument("--batch_size", type=int, default=256) 
    parser.add_argument("--save_dir", type=str, default="models/cae")
    parser.add_argument("--checkpoint_dir", type=str, default=".", help="Where to save telemetry")
    parser.add_argument("--model_name", type=str, default="texture_contrastive")
    parser.add_argument("--root_dir", type=str, default="src/images/reid-10k")
    parser.add_argument("--csv_path", type=str, default="src/images/reid-10k/metadata.csv")
    parser.add_argument("--img_size", type=int, default=256)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pth")

    print(f"=========================================")
    print(f" STARTING CONTRASTIVE CAE TRAINING")
    print(f" Target: {save_path}")
    print(f"=========================================")

    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
        train_and_save_cae(df, args, save_path, device)
    except FileNotFoundError:
        print(f"[ERROR] Could not find metadata CSV at {args.csv_path}")
        sys.exit(1)