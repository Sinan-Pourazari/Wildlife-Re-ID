import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image, ImageOps
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops
import torchvision.transforms as T
import warnings
from gnn.cae import TextureEncoder

warnings.filterwarnings("ignore")

def debug_cae_reconstruction(img_path, weights_path, latent_dim=24, img_size=256, patch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Loading CAE from {weights_path} onto {device}")
    
    # 1. Load Model 
    model = TextureEncoder(latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print("--> Weights loaded successfully.")
    except FileNotFoundError:
        print("[!] Weights not found. Visualizing with random untrained weights to test pipeline.")
    model.eval()

    # 2. Load & Pad Image (matching your pipeline exactly)
    print(f"--> Processing Image: {img_path}")
    img_pil = Image.open(img_path).convert("RGB")
    
    # Resize first if the image is massive, then pad to perfect square
    img_pil.thumbnail((img_size, img_size)) 
    img_pil = ImageOps.pad(img_pil, (img_size, img_size), color=(0, 0, 0))
    img_np = np.array(img_pil)

    # 3. Segment the image
    # Note: min_size is set slightly lower here just to ensure we get plenty of patches for the grid!
    segments = felzenszwalb(img_np, scale=70.0, sigma=0.65, min_size=60)
    regions = regionprops(segments)
    print(f"--> Found {len(regions)} superpixel segments")

    # The exact normalization the CAE was trained on
    model_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patches_for_model = []
    patch_images_for_plot = []
    
    # 4. Extract patches based on Tight Bounding Boxes
    for props in regions:
        # Get the exact bounding box of the superpixel (top, left, bottom, right)
        min_row, min_col, max_row, max_col = props.bbox
        
        # Crop the image exactly to the superpixel's boundaries
        patch_crop = img_pil.crop((min_col, min_row, max_col, max_row))
        
        # Resize that tight crop to the 64x64 size
        patch_resized = patch_crop.resize((patch_size, patch_size), Image.BILINEAR)
        
        # Save the pure, un-normalized image for the "Orig" row in the plot
        patch_images_for_plot.append(patch_resized)
        
        # Normalize the tensor for the model's forward pass
        patches_for_model.append(model_transform(patch_resized).unsqueeze(0))

    if not patches_for_model:
        print("[!] No patches extracted.")
        return

    # 5. Pass all patches through the full CAE
    batch_tensor = torch.cat(patches_for_model, dim=0).to(device)
    with torch.no_grad():
        # Your CAE returns the latent vector AND the reconstructed patches!
        reconstructed_tensors, embeddings = model(batch_tensor)

    print(f"--> Extracted {embeddings.shape[0]} embeddings.")

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    
    # Plot 1: The Segmentation Map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Padded Image")
    axes[0].axis('off')
    
    axes[1].imshow(mark_boundaries(img_np, segments))
    axes[1].set_title(f"Felzenszwalb Superpixels (N={len(regions)})")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Plot 2: High-Density Grid Reconstruction Comparison
    cols = 8  # How many patches to show side-by-side
    num_to_show = min(48, len(patch_images_for_plot)) 
    patch_rows = int(np.ceil(num_to_show / cols))
    
    # Create the grid: every row of patches needs 2 rows of subplots (Orig + Recon)
    fig, axes = plt.subplots(patch_rows * 2, cols, figsize=(cols * 1.5, patch_rows * 2 * 1.5))
    axes = np.atleast_2d(axes) # Ensure safe 2D indexing
    
    # Note: We DO NOT denormalize reconstructed_tensors here anymore!
    # The Sigmoid layer outputs perfect [0.0, 1.0] pixels naturally.
    
    for i in range(patch_rows * cols):
        r = (i // cols) * 2  # The math to alternate rows (0, 2, 4...)
        c = i % cols
        
        # Turn off axis ticks for everything
        axes[r, c].axis('off')
        axes[r+1, c].axis('off')
        
        if i < num_to_show:
            # Top Row: Original Patch (from our saved pure PIL images)
            axes[r, c].imshow(patch_images_for_plot[i])
            
            # Bottom Row: CAE Reconstruction
            recon_tensor = reconstructed_tensors[i].cpu()
            
            # 1. Denormalize using the exact inverse of your training stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            recon_tensor = (recon_tensor * std) + mean
            
            # 2. Clamp strictly to [0, 1] to make Matplotlib happy, then permute to HWC
            recon_img = recon_tensor.clamp(0, 1).permute(1, 2, 0).numpy()
            
            axes[r+1, c].imshow(recon_img)
            
            # Add clean labels only to the far-left column
            if c == 0:
                axes[r, c].text(-0.15, 0.5, 'Orig', va='center', ha='right', transform=axes[r, c].transAxes, fontsize=12, fontweight='bold', color='black')
                axes[r+1, c].text(-0.15, 0.5, 'CAE', va='center', ha='right', transform=axes[r+1, c].transAxes, fontsize=12, fontweight='bold', color='purple')

    plt.suptitle(f"CAE Texture Reconstructions (Showing {num_to_show} patches)", fontsize=16, fontweight='bold')
    
    # Squeeze the grid tightly together to maximize screen real estate
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.1, right=0.95, wspace=0.05, hspace=0.1)
    plt.show()

if __name__ == "__main__":
    # ---> CHANGE THESE PATHS TO MATCH YOUR LOCAL SETUP <---
    TEST_IMAGE = r"src\ml_utils\gnn\000011.jpg"
    
    # Point this directly to your newly trained CAE weights
    CAE_WEIGHTS = r"C:\Users\sinan\Projects\Wildlife-Re-ID\models\cae\cae_dim32_size256_scale70p0_sigma0p65_clef_big_v2.pth"
    
    debug_cae_reconstruction(
        img_path=TEST_IMAGE, 
        weights_path=CAE_WEIGHTS,
        latent_dim=32,  # Make sure this matches what you pre-trained with
        img_size=256,   
        patch_size=64   
    )