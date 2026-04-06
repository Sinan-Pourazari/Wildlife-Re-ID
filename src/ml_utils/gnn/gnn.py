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

def image_to_superpixel_graph(img, mask=None, n_segments=300, hog_bins=9, hog_signed=False, hog_l2norm=True, features=['color', 'pos', 'hog']):
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, _ = img.shape

    # --- 1. Standard SLIC ---
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)
    num_nodes = segments.max() + 1
    seg_flat = torch.tensor(segments, dtype=torch.long).view(-1)
    
    x_list = []

    # --- 2. Color, Position, and HOG (Original Loop Logic) ---
    if any(f in features for f in ['color', 'pos', 'hog']):
        img_lab = rgb2lab(img)
        if 'hog' in features:
            pix_bin_idx, pix_mag = per_pixel_hog_bins(img, n_bins=hog_bins, signed=hog_signed)

        colors, positions, hogs = [], [], []

        for sp in range(num_nodes):
            mask_sp = (segments == sp)
            coords = np.column_stack(np.nonzero(mask_sp))
            vals = img_lab[mask_sp]

            if 'color' in features:
                L, A, B = vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()
                colors.append([L, A, B])

            if 'pos' in features:
                y, x = coords[:, 0].mean(), coords[:, 1].mean()
                positions.append([x / w, y / h])

            if 'hog' in features:
                sp_bins = pix_bin_idx[mask_sp].ravel()
                sp_w = pix_mag[mask_sp].ravel()
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
        lbp = local_binary_pattern(gray, P=8, R=1.0, method='uniform')
        lbp_flat = torch.tensor(lbp, dtype=torch.long).view(-1)
        
        lbp_one_hot = F.one_hot(lbp_flat, num_classes=10).float()
        x_lbp = torch.zeros((num_nodes, 10), dtype=torch.float)
        x_lbp.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, 10), lbp_one_hot)
        
        x_lbp = F.normalize(x_lbp, p=1, dim=1) 
        x_list.append(x_lbp)

    # --- 4. NEW: Local Entropy (Chaos/Smoothness metric) ---
    if 'texture' in features:
        gray_uint8 = (rgb2gray(img) * 255).astype(np.uint8)
        ent = entropy(gray_uint8, disk(3))
        ent_flat = torch.tensor(ent, dtype=torch.float).view(-1, 1)
        
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).view(-1, 1).float()
        x_ent = torch.zeros((num_nodes, 1), dtype=torch.float)
        x_ent.scatter_add_(0, seg_flat.unsqueeze(1), ent_flat)
        x_ent = x_ent / (node_counts + 1e-6)
        x_list.append(x_ent)

    # --- Combine all selected features ---
    x = torch.cat(x_list, dim=1)

    # ==========================================
    # --- 5. GRAPH PRUNING (Using the Mask) ---
    # ==========================================
    if mask is not None:
        mask_flat = torch.tensor(mask, dtype=torch.float).view(-1)
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).float()
        
        node_mask_scores = torch.zeros(num_nodes, dtype=torch.float)
        node_mask_scores.scatter_add_(0, seg_flat, mask_flat)
        node_mask_scores = node_mask_scores / (node_counts + 1e-6)

        is_valid_node = node_mask_scores > 127.0 

        x_pruned = x[is_valid_node]

        old_to_new_ids = torch.full((num_nodes,), -1, dtype=torch.long)
        old_to_new_ids[is_valid_node] = torch.arange(x_pruned.size(0))

        # --- Build Pruned Edges ---
        edges = set()
        for y in range(h - 1):
            for x_ in range(w - 1):
                a = segments[y, x_]
                b = segments[y, x_ + 1]
                c = segments[y + 1, x_]

                if a != b and is_valid_node[a] and is_valid_node[b]:
                    edges.add((old_to_new_ids[a].item(), old_to_new_ids[b].item()))
                    edges.add((old_to_new_ids[b].item(), old_to_new_ids[a].item()))
                    
                if a != c and is_valid_node[a] and is_valid_node[c]:
                    edges.add((old_to_new_ids[a].item(), old_to_new_ids[c].item()))
                    edges.add((old_to_new_ids[c].item(), old_to_new_ids[a].item()))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        data = Data(x=x_pruned, edge_index=edge_index)
        
    else:
        # --- Standard Edge Building (If no mask is provided) ---
        edges = set()
        for y in range(h - 1):
            for x_ in range(w - 1):
                a = segments[y, x_]
                b = segments[y, x_ + 1]
                c = segments[y + 1, x_]

                if a != b:
                    edges.add((a, b))
                    edges.add((b, a))
                if a != c:
                    edges.add((a, c))
                    edges.add((c, a))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        
    return data

# ==========================================
# --- Restored GNN Encoder ---
# ==========================================
class GNNEncoder(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=512, out_dim=512):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim//8, heads=8)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, out_dim // 8, heads=8)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x = self.norm1(F.elu(self.conv1(x, edge_index)))

        # Layer 2 (with Residual)
        identity = x
        x = self.norm2(F.elu(self.conv2(x, edge_index)))
        x = x + identity

        # Layer 3
        x = self.norm3(F.elu(self.conv3(x, edge_index)))

        # Pooling
        pooled_mean = global_mean_pool(x, batch) 
        pooled_max = global_max_pool(x, batch)   
        
        return torch.cat([pooled_mean, pooled_max], dim=1)