import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image, ImageOps
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
import torchvision.transforms as T
import warnings
from gnn.cae import TextureEncoder
import cv2
from cv2.ximgproc import createSuperpixelSEEDS

warnings.filterwarnings("ignore")

def color_average_segments(image, segments):
    """Return an image where each superpixel is replaced by its mean colour."""
    h, w, c = image.shape
    avg_img = np.zeros_like(image, dtype=np.uint8)
    for region in regionprops(segments + 1):
        label = region.label - 1
        mask = (segments == label)
        if mask.any():
            mean_color = image[mask].mean(axis=0).astype(np.uint8)
            avg_img[mask] = mean_color
    return avg_img

def compute_adjacency_and_centroids(segments):
    """Return adjacency list (list of (i,j) tuples) and centroids (list of (x,y))."""
    h, w = segments.shape
    labels = np.unique(segments)
    n = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    centroids = [None] * n
    for region in regionprops(segments + 1):
        label = region.label - 1
        idx = label_to_idx[label]
        centroids[idx] = (region.centroid[1], region.centroid[0])   # (x, y) for plotting
    # Build adjacency set
    adj = set()
    for y in range(h - 1):
        for x in range(w - 1):
            l = segments[y, x]
            r = segments[y, x+1]
            if l != r:
                adj.add(tuple(sorted((label_to_idx[l], label_to_idx[r]))))
            d = segments[y+1, x]
            if l != d:
                adj.add(tuple(sorted((label_to_idx[l], label_to_idx[d]))))
    return list(adj), centroids

def process_image(img_path, output_name, model, device, img_size=512, patch_size=64, skip_graph=False, save_this=False, root_folder_name="graph_outputs"):
    """Process a single image.
    
    Args:
        skip_graph: If True, do not draw edges and nodes.
        save_this: If True, save the graph overview; otherwise skip saving.
        root_folder_name: Folder inside root where to save (only if save_this=True).
    """
    print(f"\n--- Processing: {img_path} -> {output_name} (skip_graph={skip_graph}, save_this={save_this}) ---")
    
    # Load and pad image
    img_pil = Image.open(img_path).convert("RGB")
    img_pil.thumbnail((img_size, img_size))
    img_pil = ImageOps.pad(img_pil, (img_size, img_size), color=(0, 0, 0))
    img_np = np.array(img_pil)

    # SEEDS superpixel segmentation
    h, w, c = img_np.shape
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    target_superpixels = 240
    seeds_algo = createSuperpixelSEEDS(w, h, c, target_superpixels, num_levels=4, prior=1, histogram_bins=4)
    seeds_algo.iterate(img_hsv, 4)
    segments = seeds_algo.getLabels()
    regions = list(regionprops(segments + 1))
    print(f"--> Found {len(regions)} superpixel segments")

    # Build colour‑averaged image and graph
    avg_img = color_average_segments(img_np, segments)
    if not skip_graph:
        edges, centroids = compute_adjacency_and_centroids(segments)
    else:
        edges, centroids = [], []

    # ========== FIGURE 1: Two‑panel overview ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("Original (padded)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Colour‑averaged + superpixel outlines
    avg_with_boundaries = mark_boundaries(avg_img, segments, color=(1, 1, 0), mode='thick')
    axes[1].imshow(avg_with_boundaries)
    
    # Draw graph connections only if not skipped
    if not skip_graph:
        for (i, j) in edges:
            xi, yi = centroids[i]
            xj, yj = centroids[j]
            axes[1].plot([xi, xj], [yi, yj], 'w-', linewidth=2.5, alpha=0.95, solid_capstyle='round')
        for idx, (cx, cy) in enumerate(centroids):
            area = regions[idx].area
            size = max(12, min(80, int(area / 80)))
            axes[1].scatter(cx, cy, s=size, c='#00ccff', edgecolor='white', linewidth=1.5, alpha=0.95, zorder=10)
    
    title = "Colour‑averaged + superpixel outlines" + ("" if skip_graph else " + graph")
    axes[1].set_title(title, fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout(pad=0.5)
    
    # Only save if save_this is True
    if save_this:
        root_dir = os.getcwd()
        output_folder = os.path.join(root_dir, root_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{output_name}_graph_overview.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"--> Saved graph overview to {output_path}")
    else:
        print(f"--> Skipping save for {output_name}")
    
    plt.close(fig)

    # ========== FIGURE 2: CAE patch reconstruction grid (always shown) ==========
    model_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patches_for_model = []
    patch_images_for_plot = []

    for props in regions:
        min_row, min_col, max_row, max_col = props.bbox
        segment_id = props.label - 1
        mask_sp = (segments == segment_id)
        crop_np = img_np[min_row:max_row, min_col:max_col].copy()
        local_mask = mask_sp[min_row:max_row, min_col:max_col]
        if local_mask.any():
            mean_color = crop_np[local_mask].mean(axis=0).astype(np.uint8)
        else:
            mean_color = np.array([0, 0, 0], dtype=np.uint8)
        crop_np[~local_mask] = mean_color
        patch_crop = Image.fromarray(crop_np)
        patch_resized = patch_crop.resize((patch_size, patch_size), Image.BILINEAR)
        patch_images_for_plot.append(patch_resized)
        patches_for_model.append(model_transform(patch_resized).unsqueeze(0))

    if not patches_for_model:
        print("[!] No patches extracted.")
        return

    batch_tensor = torch.cat(patches_for_model, dim=0).to(device)
    with torch.no_grad():
        reconstructed_tensors, embeddings = model(batch_tensor)
    print(f"--> Extracted {embeddings.shape[0]} embeddings.")

    # Plot reconstruction grid
    cols = 8
    num_to_show = min(48, len(patch_images_for_plot))
    patch_rows = int(np.ceil(num_to_show / cols))
    fig2, axes2 = plt.subplots(patch_rows * 2, cols, figsize=(cols * 1.5, patch_rows * 2 * 1.5))
    axes2 = np.atleast_2d(axes2)
    for i in range(patch_rows * cols):
        r = (i // cols) * 2
        c = i % cols
        axes2[r, c].axis('off')
        axes2[r+1, c].axis('off')
        if i < num_to_show:
            axes2[r, c].imshow(patch_images_for_plot[i])
            recon_tensor = reconstructed_tensors[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            recon_tensor = (recon_tensor * std) + mean
            recon_img = recon_tensor.clamp(0, 1).permute(1, 2, 0).numpy()
            axes2[r+1, c].imshow(recon_img)
            if c == 0:
                axes2[r, c].text(-0.15, 0.5, 'Masked Orig', va='center', ha='right', transform=axes2[r, c].transAxes, fontsize=12, fontweight='bold')
                axes2[r+1, c].text(-0.15, 0.5, 'CAE', va='center', ha='right', transform=axes2[r+1, c].transAxes, fontsize=12, fontweight='bold', color='purple')
    plt.suptitle(f"CAE Masked Texture Reconstructions ({num_to_show} patches)", fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.1, right=0.95, wspace=0.05, hspace=0.1)
    plt.show()
    plt.close(fig2)

def main():
    CAE_WEIGHTS = r"C:\Users\sinan\Projects\Wildlife-Re-ID\models\cae\cae_dim90_size512_seeds240_v2.pth"
    latent_dim = 90
    img_size = 512
    patch_size = 64

    device = torch.device('cpu')
    model = TextureEncoder(latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(torch.load(CAE_WEIGHTS, map_location=device), strict=False)
        print("--> CAE weights loaded successfully.")
    except FileNotFoundError:
        print("[!] Weights not found. Using random untrained weights.")
    model.eval()

    # List of (image_path, output_name, skip_graph, save_this)
    # Only salamander will save the graph overview (skip_graph=True, save_this=True)
    images = [
        (r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\images\LynxID2025\test\2f2432eb73762a67711508c2a92e2004f091f23d03888fccabeffd132083f51e.jpg", "lynx", True, True),
        (r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\images\SalamanderID2025\test\0bb7bedeb8123132_95.jpg", "salamander", True, True),
        (r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\images\SeaTurtleID2022\test\0d942316aeb78978_23.JPG", "turtle", True, True),
        (r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\animal-clef-2026\images\TexasHornedLizards\test\1e177a6eab060e92.jpg", "texas_lizard", True, True)
    ]

    for img_path, out_name, skip_graph, save_this in images:
        if not os.path.exists(img_path):
            print(f"[!] File not found: {img_path}")
            continue
        process_image(img_path, out_name, model, device, img_size, patch_size, skip_graph, save_this)

if __name__ == "__main__":
    main()