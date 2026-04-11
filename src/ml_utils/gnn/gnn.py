import torch
import numpy as np
from skimage.segmentation import slic, felzenszwalb
from skimage.color import rgb2lab, rgb2gray
from skimage.io import imread
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, knn_graph
from PIL import Image
from helper import per_pixel_hog_bins # Make sure this import is still correct for your project!
from gnn.cae import TextureEncoder
from torchvision import transforms
from skimage.measure import regionprops
import math
def image_to_superpixel_graph(img, mask=None, n_segments=300, hog_bins=9, hog_signed=False, hog_l2norm=True, features=['color', 'pos', 'hog'], n_hops=1, cae_weights_path=None, cae_latent_dim=64, return_segments=False):   
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, _ = img.shape

    # --- 1. Standard SLIC ---
    segments = felzenszwalb(img, scale=70.0, sigma=0.65, min_size=150)
    num_nodes = segments.max() + 1
    seg_flat = torch.tensor(segments, dtype=torch.long).view(-1)
    
    x_list = []

# --- CAE SETUP ---
    if 'cae' in features:
        if cae_weights_path is None:
            raise ValueError("Requested 'cae' features but no cae_weights_path provided!")
        cae_model = get_cae_model(cae_weights_path, cae_latent_dim)
        
        cae_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)), # <--- CHANGE THIS FROM 64 TO 32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # --- 2. Color, Position, HOG, and CAE Loop ---
    if any(f in features for f in ['color', 'pos', 'hog', 'cae', 'shape']):
        img_lab = rgb2lab(img)
        if 'hog' in features:
            pix_bin_idx, pix_mag = per_pixel_hog_bins(img, n_bins=hog_bins, signed=hog_signed)
        
        if 'shape' in features:
            props = regionprops(segments + 1)
            total_area = h * w
            max_perimeter = 2 * (h + w)
        colors, positions, hogs, cae_features, shape_features = [], [], [], [], []

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
                
            # CAE Feature Extraction ---
            if 'cae' in features:
                # 1. Find the Bounding Box of the superpixel
                ymin, ymax = coords[:, 0].min(), coords[:, 0].max()
                xmin, xmax = coords[:, 1].min(), coords[:, 1].max()
                
                # 2. Crop the original RGB image to that box
                crop_img = img[ymin:ymax+1, xmin:xmax+1]
                
                # 3. Apply the transform and add batch dimension [1, C, H, W]
                crop_tensor = cae_transform(crop_img).unsqueeze(0)
                
                # 4. Push through the encoder
                with torch.no_grad():
                    latent_vector = cae_model.encoder(crop_tensor).squeeze(0)
                    
                cae_features.append(latent_vector.numpy())
                
            if 'shape' in features:
                prop = props[sp]
                
                area = prop.area / total_area
                perimeter = prop.perimeter / max_perimeter
                
                min_y, min_x, max_y, max_x = prop.bbox
                bb_h = max(max_y - min_y, 1)
                bb_w = max(max_x - min_x, 1)
                aspect_ratio = bb_w / bb_h
                
                circularity = (4 * math.pi * prop.area) / ((prop.perimeter ** 2) + 1e-6)
                solidity = prop.solidity
                extent = prop.extent
                eccentricity = prop.eccentricity
                
                hu_moments = prop.moments_hu
                log_hu = []
                for hu in hu_moments:
                    val = -1 * math.copysign(1.0, hu) * math.log10(abs(hu) + 1e-6)
                    log_hu.append(val)
                    
                node_shape_vec = [
                    area, perimeter, aspect_ratio, circularity, 
                    solidity, extent, eccentricity
                ] + log_hu
                
                shape_features.append(node_shape_vec)

        # Append extracted loop features to main list
        if 'color' in features:
            x_list.append(torch.tensor(np.array(colors), dtype=torch.float))
        if 'pos' in features:
            x_list.append(torch.tensor(np.array(positions), dtype=torch.float))
        if 'hog' in features:
            x_list.append(torch.tensor(np.array(hogs), dtype=torch.float))
        if 'cae' in features:
            x_list.append(torch.tensor(np.array(cae_features), dtype=torch.float))
        if 'shape' in features:
            x_shape = torch.tensor(np.array(shape_features), dtype=torch.float)
            # L2 Normalize the whole column to ensure stability alongside other features
            x_shape = F.normalize(x_shape, p=2, dim=0) 
            x_list.append(x_shape)

    # --- 3. Local Binary Patterns (LBP) Histogram ---
    if 'lbp' in features:
        # Multiply by 255 and convert to 8-bit integer to safely calculate LBP
        gray = (rgb2gray(img) * 255).astype(np.uint8)
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
        
    if n_hops > 1:
        actual_nodes = data.x.size(0)
        
        # 1. Create a dense adjacency matrix
        adj = torch.zeros((actual_nodes, actual_nodes), dtype=torch.float)
        adj[data.edge_index[0], data.edge_index[1]] = 1.0
        
        # 2. Add self-loops (This ensures 1-hop edges aren't lost when finding 2-hop edges)
        adj.fill_diagonal_(1.0)
        
        # 3. Multiply matrix by itself 'n' times
        adj_n = torch.matrix_power(adj, n_hops)
        
        # 4. Remove self-loops (GNNs handle this internally usually, but best to be clean)
        adj_n.fill_diagonal_(0.0)
        
        # 5. Convert back to PyG edge_index format
        new_edge_index = (adj_n > 0).nonzero(as_tuple=False).t().contiguous()
        data.edge_index = new_edge_index
        
    # Return the map if viz.py asks for it
    if return_segments:
        return data, segments
        
    return data


class GNNEncoder(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=512, out_dim=512, edge_strategy="spatial", k_neighbors=1):
        super().__init__()
        # The number of attention-based semantic edges to create per node
        self.k_neighbors = k_neighbors
        self.edge_strategy = edge_strategy
        # Learned Attention Projection
        # Projects raw features into a specific "Edge Similarity" space.
        # This acts like the Query/Key transformations in standard Transformers.
        if self.edge_strategy in ["attention", "hybrid"]:
            self.edge_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
        self.conv1 = GATv2Conv(in_dim, hidden_dim//8, heads=8)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, out_dim // 8, heads=8)
        self.norm3 = nn.LayerNorm(out_dim)
        
        # Initialize the GeM Pooling layer here
        self.gem_pool = GeMPooling(p=3.0)

    def forward(self, data):
        x, spatial_edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.edge_strategy == "spatial":
            # Baseline: Use only the CPU-generated LMDB edges
            final_edge_index = spatial_edge_index
            
        else:
            # Generate Attention Edges
            queries_keys = self.edge_proj(x)
            semantic_edge_index = knn_graph(
                x=queries_keys, k=self.k_neighbors, batch=batch, loop=False, cosine=True
            )
            
            if self.edge_strategy == "attention":
                # Pure semantic approach (ignores physical layout)
                final_edge_index = semantic_edge_index
            elif self.edge_strategy == "hybrid":
                # Best of both worlds: Anatomy + Texture Matching
                final_edge_index = torch.cat([spatial_edge_index, semantic_edge_index], dim=1)
        # Layer 1
        x = self.norm1(F.elu(self.conv1(x, final_edge_index)))

        # Layer 2 (with Residual)
        identity = x
        x = self.norm2(F.elu(self.conv2(x, final_edge_index)))
        x = x + identity

        # Layer 3
        x = self.norm3(F.elu(self.conv3(x, final_edge_index)))

        # Pooling
        #pooled_mean = global_mean_pool(x, batch) 
        #pooled_max = global_max_pool(x, batch)   
        pooled = self.gem_pool(x, batch)
        return pooled
        #return torch.cat([pooled_mean, pooled_max], dim=1)
    
class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        # 'p' is instantiated as a learnable PyTorch Parameter.
        # It initializes at 3.0 but the optimizer will update it during training.
        self.p = nn.Parameter(torch.ones(1) * p) 
        self.eps = eps

    def forward(self, x, batch):
        # 1. Clamp to ensure all values are strictly positive
        x = x.clamp(min=self.eps)
        
        # 2. Raise all features to the power of p
        x = x.pow(self.p)
        
        # 3. Perform standard Average Pooling (accounting for graph batches)
        x_pool = global_mean_pool(x, batch)
        
        # 4. Take the p-th root
        x_pool = x_pool.pow(1.0 / self.p)
        
        return x_pool
    
_WORKER_CAE_CACHE = {}  
def get_cae_model(weights_path, latent_dim):
    """Loads the CAE once per worker process and keeps it in RAM."""
    global _WORKER_CAE_CACHE
    if weights_path not in _WORKER_CAE_CACHE:
        # Pass the dynamic latent_dim to the architecture
        model = TextureEncoder(latent_dim=latent_dim) 
        
        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        _WORKER_CAE_CACHE[weights_path] = model
        
    return _WORKER_CAE_CACHE[weights_path]