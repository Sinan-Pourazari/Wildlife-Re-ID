import torch
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab, rgb2gray
from skimage.io import imread
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from PIL import Image
from helper import per_pixel_hog_bins # Make sure this import is still correct for your project!

def image_to_superpixel_graph(img, n_segments=300, hog_bins=9, hog_signed=False, hog_l2norm=True, features=['color', 'pos', 'hog']):
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, _ = img.shape

    # --- 1. Standard SLIC ---
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)
    num_nodes = segments.max() + 1
    seg_flat = torch.tensor(segments, dtype=torch.long).view(-1)
    
    x_list = []

    # --- 2. Color, Position, and HOG (Original Loop Logic) ---
    # We only run this loop if at least one of these features is requested
    if any(f in features for f in ['color', 'pos', 'hog']):
        img_lab = rgb2lab(img)
        if 'hog' in features:
            pix_bin_idx, pix_mag = per_pixel_hog_bins(img, n_bins=hog_bins, signed=hog_signed)

        colors, positions, hogs = [], [], []

        for sp in range(num_nodes):
            mask = (segments == sp)
            coords = np.column_stack(np.nonzero(mask))
            vals = img_lab[mask]

            if 'color' in features:
                L, A, B = vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()
                colors.append([L, A, B])

            if 'pos' in features:
                y, x = coords[:, 0].mean(), coords[:, 1].mean()
                positions.append([x / w, y / h])

            if 'hog' in features:
                sp_bins = pix_bin_idx[mask].ravel()
                sp_w = pix_mag[mask].ravel()
                sp_bins = np.clip(sp_bins, 0, hog_bins - 1)
                hog_hist = np.bincount(sp_bins, weights=sp_w, minlength=hog_bins).astype(np.float32)[:hog_bins]
                if hog_l2norm:
                    hog_hist /= (np.linalg.norm(hog_hist, ord=2) + 1e-6)
                hogs.append(hog_hist)

        if 'color' in features:
            x_list.append(torch.tensor(np.array(colors), dtype=torch.float))
        if 'pos' in features:
            x_list.append(torch.tensor(np.array(positions), dtype=torch.float))
        if 'hog' in features:
            x_list.append(torch.tensor(np.array(hogs), dtype=torch.float))

    # --- 3. NEW: Local Binary Patterns (LBP) Histogram ---
    if 'lbp' in features:
        gray = rgb2gray(img)
        # P=8, R=1.0 uniform gives 10 distinct texture patterns (edges, corners, flats)
        lbp = local_binary_pattern(gray, P=8, R=1.0, method='uniform')
        lbp_flat = torch.tensor(lbp, dtype=torch.long).view(-1)
        
        # Fast Vectorized Histogram pooling per superpixel
        lbp_one_hot = F.one_hot(lbp_flat, num_classes=10).float() # Shape: [Pixels, 10]
        x_lbp = torch.zeros((num_nodes, 10), dtype=torch.float)
        x_lbp.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, 10), lbp_one_hot)
        
        # L1 Normalize to turn raw counts into a percentage frequency distribution
        x_lbp = F.normalize(x_lbp, p=1, dim=1) 
        x_list.append(x_lbp)

    # --- 4. NEW: Local Entropy (Chaos/Smoothness metric) ---
    if 'texture' in features:
        gray_uint8 = (rgb2gray(img) * 255).astype(np.uint8)
        # Calculate local entropy using a small 3-pixel radius
        ent = entropy(gray_uint8, disk(3))
        ent_flat = torch.tensor(ent, dtype=torch.float).view(-1, 1)
        
        # Fast mean pooling per superpixel
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).view(-1, 1).float()
        x_ent = torch.zeros((num_nodes, 1), dtype=torch.float)
        x_ent.scatter_add_(0, seg_flat.unsqueeze(1), ent_flat)
        x_ent = x_ent / (node_counts + 1e-6) # Divide sum by pixel count to get mean
        x_list.append(x_ent)

# --- Combine all selected features ---
    x = torch.cat(x_list, dim=1)
    
    # ==========================================
    # --- 5. GRAPH PRUNING (Kill the Padding) ---
    # ==========================================
    
    # 1. Calculate the average RGB color of every superpixel
    # We use PyTorch scatter to do this instantly on the CPU
    img_flat = torch.tensor(img, dtype=torch.float).view(-1, 3)
    node_counts = torch.bincount(seg_flat, minlength=num_nodes).view(-1, 1).float()
    
    node_colors = torch.zeros((num_nodes, 3), dtype=torch.float)
    node_colors.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, 3), img_flat)
    node_colors = node_colors / (node_counts + 1e-6)

    # 2. A node is "padding" if its average color is pure black (Sum of RGB is tiny)
    # 5.0 out of 765 gives a tiny buffer for jpeg compression artifacts
    is_valid_node = node_colors.sum(dim=1) > 5.0  

    # 3. Filter the features: Keep only the valid nodes! 
    # This shrinks x from e.g., [300, in_dim] to [210, in_dim]
    x_pruned = x[is_valid_node]

    # 4. Create a mapping array to fix the edge connections
    # If we delete node 5, the old node 6 needs to become the new node 5.
    old_to_new_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new_ids[is_valid_node] = torch.arange(x_pruned.size(0))

    # ==========================================
    # --- 6. Build Pruned Edges ---
    # ==========================================
    edges = set()
    for y in range(h - 1):
        for x_ in range(w - 1):
            a = segments[y, x_]
            b = segments[y, x_ + 1]
            c = segments[y + 1, x_]

            # Only add the edge if BOTH nodes are valid (not black padding)
            if a != b and is_valid_node[a] and is_valid_node[b]:
                # Map the old segment IDs to the new pruned IDs
                new_a, new_b = old_to_new_ids[a].item(), old_to_new_ids[b].item()
                edges.add((new_a, new_b))
                edges.add((new_b, new_a))
                
            if a != c and is_valid_node[a] and is_valid_node[c]:
                new_a, new_c = old_to_new_ids[a].item(), old_to_new_ids[c].item()
                edges.add((new_a, new_c))
                edges.add((new_c, new_a))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # Pass the pruned features (x_pruned) instead of the raw x
    data = Data(x=x_pruned, edge_index=edge_index)
    return data