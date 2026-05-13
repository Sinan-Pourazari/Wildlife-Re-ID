import torch
import numpy as np
import cv2
import math
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import regionprops
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node, from_scipy_sparse_matrix
from helper import per_pixel_hog_bins
from gnn.cae import TextureEncoder
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, knn_graph, global_max_pool

def image_to_superpixel_graph(img, seeds_num_superpixels=300, seeds_num_levels=4, seeds_prior=1, seeds_histogram_bins=4, hog_bins=9, mask=None, hog_signed=False, hog_l2norm=True, features=['color', 'pos', 'hog'], n_hops=1, cae_weights_path=None, cae_latent_dim=128, return_segments=False):   
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, c = img.shape

    # ==================================================================
    # 1. Instant Segmentation via SEEDS
    # ==================================================================
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    seeds_algo = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, c, 
        seeds_num_superpixels, 
        num_levels=seeds_num_levels, 
        prior=seeds_prior, 
        histogram_bins=seeds_histogram_bins
    )
    
    try:
        seeds_algo.iterate(img_hsv, 4)
    except cv2.error:
        # C++ MEMORY CRASH PREVENTER: If image math fails, fallback to 1 level
        # print(f"[!] SEEDS math failed on shape {img.shape}. Applying safe fallback.")
        seeds_algo = cv2.ximgproc.createSuperpixelSEEDS(
            w, h, c, seeds_num_superpixels, num_levels=1, prior=seeds_prior, histogram_bins=seeds_histogram_bins
        )
        seeds_algo.iterate(img_hsv, 4)
        
    raw_segments = seeds_algo.getLabels()
    
    # CRITICAL FIX: Ensure labels are strictly contiguous (SEEDS sometimes skips IDs)
    unique_labels, segments = np.unique(raw_segments, return_inverse=True)
    segments = segments.reshape(h, w)
    
    num_nodes = len(unique_labels)
    seg_flat = torch.tensor(segments, dtype=torch.long).view(-1)
    
    x_list = []

    # ==================================================================
    # 2. Extract Features (Vectorized + Batched)
    # ==================================================================
    if any(f in features for f in ['color', 'pos', 'hog', 'cae', 'shape']):
        img_lab = rgb2lab(img)
        if 'hog' in features:
            pix_bin_idx, pix_mag = per_pixel_hog_bins(img, n_bins=hog_bins, signed=hog_signed)
        
        props = regionprops(segments + 1)
        total_area = h * w
        max_perimeter = 2 * (h + w)
            
        colors, positions, hogs, shape_features = [], [], [], []
        
        if 'cae' in features:
            cae_model = get_cae_model(cae_weights_path, cae_latent_dim)
            cae_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            all_crops = [None] * num_nodes 

        for sp_idx in range(num_nodes):
            prop = props[sp_idx]
            min_y, min_x, max_y, max_x = prop.bbox
            
            # O(1) Masking
            local_mask = prop.image 
            vals = img_lab[min_y:max_y, min_x:max_x][local_mask]

            if 'color' in features:
                L, A, B = vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()
                colors.append([L, A, B])

            if 'pos' in features:
                y, x = prop.centroid
                positions.append([x / w, y / h])

            if 'hog' in features:
                sp_bins = pix_bin_idx[min_y:max_y, min_x:max_x][local_mask].ravel()
                sp_w = pix_mag[min_y:max_y, min_x:max_x][local_mask].ravel()
                sp_bins = np.clip(sp_bins, 0, hog_bins - 1)
                hog_hist = np.bincount(sp_bins, weights=sp_w, minlength=hog_bins).astype(np.float32)[:hog_bins]
                if hog_l2norm:
                    hog_hist /= (np.linalg.norm(hog_hist, ord=2) + 1e-6)
                hogs.append(hog_hist)
                
            if 'cae' in features:
                crop_img = img[min_y:max_y, min_x:max_x].copy()
                
                # MEAN MASKING
                if local_mask.any():
                    mean_color = crop_img[local_mask].mean(axis=0).astype(np.uint8)
                else:
                    mean_color = np.array([0, 0, 0], dtype=np.uint8)
                crop_img[~local_mask] = mean_color
                
                all_crops[sp_idx] = cae_transform(crop_img)
                
            if 'shape' in features:
                area = prop.area / total_area
                perimeter = prop.perimeter / max_perimeter
                bb_h = max(max_y - min_y, 1)
                bb_w = max(max_x - min_x, 1)
                circularity = (4 * math.pi * prop.area) / ((prop.perimeter ** 2) + 1e-6)
                log_hu = [-1 * math.copysign(1.0, hu) * math.log10(abs(hu) + 1e-6) for hu in prop.moments_hu]
                shape_features.append([area, perimeter, bb_w / bb_h, circularity, prop.solidity, prop.extent, prop.eccentricity] + log_hu)

        # Batched CAE Inference
        if 'cae' in features:
            batch_tensor = torch.stack(all_crops)
            cae_device = next(cae_model.parameters()).device
            with torch.no_grad():
                cae_features = cae_model.encoder(batch_tensor.to(cae_device)).cpu().numpy()
            x_list.append(torch.tensor(cae_features, dtype=torch.float))

        if 'color' in features: x_list.append(torch.tensor(np.array(colors), dtype=torch.float))
        if 'pos' in features: x_list.append(torch.tensor(np.array(positions), dtype=torch.float))
        if 'hog' in features: x_list.append(torch.tensor(np.array(hogs), dtype=torch.float))
        if 'shape' in features:
            x_shape = torch.tensor(np.array(shape_features), dtype=torch.float)
            x_list.append(F.normalize(x_shape, p=2, dim=0))

    if 'lbp' in features:
        gray = (rgb2gray(img) * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray, P=8, R=1.0, method='uniform')
        lbp_flat = torch.tensor(lbp, dtype=torch.long).view(-1)
        lbp_one_hot = F.one_hot(lbp_flat, num_classes=10).float()
        x_lbp = torch.zeros((num_nodes, 10), dtype=torch.float)
        x_lbp.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, 10), lbp_one_hot)
        x_list.append(F.normalize(x_lbp, p=1, dim=1))

    if 'texture' in features:
        gray_uint8 = (rgb2gray(img) * 255).astype(np.uint8)
        ent = entropy(gray_uint8, disk(3))
        ent_flat = torch.tensor(ent, dtype=torch.float).view(-1, 1)
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).view(-1, 1).float()
        x_ent = torch.zeros((num_nodes, 1), dtype=torch.float)
        x_ent.scatter_add_(0, seg_flat.unsqueeze(1), ent_flat)
        x_list.append(x_ent / (node_counts + 1e-6))

    x_raw = torch.cat(x_list, dim=1).to(torch.float32)
    x = F.layer_norm(x_raw, x_raw.shape[1:]).to(torch.bfloat16)

    # ==================================================================
    # 3. Vectorized Edge Building
    # ==================================================================
    v_edges = np.column_stack((segments[:-1, :].ravel(), segments[1:, :].ravel()))
    h_edges = np.column_stack((segments[:, :-1].ravel(), segments[:, 1:].ravel()))
    
    all_edges = np.vstack((v_edges, h_edges))
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
    
    unique_edges = np.unique(all_edges, axis=0)
    edges_bi = np.vstack((unique_edges, unique_edges[:, [1, 0]]))
    edges = np.unique(edges_bi, axis=0)

    if len(edges) > 0:
        edge_index = torch.from_numpy(edges).long().t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # ==================================================================
    # 4. Local Edge Attributes
    # ==================================================================
    edge_attr = None
    if 'pos' in features:
        if edge_index.numel() > 0:
            feature_block_sizes = {'color': 3, 'pos': 2, 'hog': hog_bins, 'shape': 14, 'lbp': 10, 'texture': 1, 'cae': cae_latent_dim}
            extraction_order = ['color', 'pos', 'hog', 'cae', 'shape', 'lbp', 'texture']
            
            pos_start = 0
            for feat in extraction_order:
                if feat == 'pos': break
                if feat in features: pos_start += feature_block_sizes[feat]
                    
            pos_tensor = x[:, pos_start : pos_start+2].to(torch.float32)
            row, col = edge_index
            
            rel_pos = pos_tensor[row] - pos_tensor[col]
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)
            angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
            sin_angle = torch.sin(angle).unsqueeze(-1)
            cos_angle = torch.cos(angle).unsqueeze(-1)
            
            edge_attr = torch.cat([dist, sin_angle, cos_angle], dim=-1).to(torch.bfloat16)
        else:
            edge_attr = torch.empty((0, 3), dtype=torch.bfloat16)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    if n_hops > 1:
        actual_nodes = data.x.size(0)
        adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(actual_nodes, actual_nodes))
        adj = adj + sp.eye(actual_nodes)
        adj_n = adj ** n_hops
        adj_n.setdiag(0)
        adj_n.eliminate_zeros()
        data.edge_index, _ = from_scipy_sparse_matrix(adj_n)
        
    if return_segments:
        return data, segments
        
    return data

    
class GNNEncoder(nn.Module):
    def __init__(self, features, cae_latent_dim, num_hog_bins, in_dim=14, hidden_dim=512, out_dim=512, 
                 edge_strategy="spatial", k_neighbors=5, gnn_layers=3, gnn_type="gatv2", 
                 use_jk=True, jk_mode="attention", pooling_type="gem", drop_node=0.10, drop_edge=0.15, drop_modality=0.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.edge_strategy = edge_strategy
        self.features = features
        self.cae_latent_dim = cae_latent_dim
        self.num_hog_bins = num_hog_bins
        self.gnn_layers = gnn_layers
        self.gnn_type = gnn_type
        self.use_jk = use_jk
        self.jk_mode = jk_mode
        self.drop_node = drop_node
        self.drop_edge = drop_edge
        self.drop_modality = drop_modality
        
        if self.edge_strategy in ["attention", "hybrid"]:
            self.edge_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
        self.edge_dim = 3 if 'pos' in features else None
        
        # --- DYNAMIC LAYER BUILDING ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        current_in_dim = in_dim
        for i in range(self.gnn_layers):
            if self.gnn_type == "gatv2":
                self.convs.append(GATv2Conv(current_in_dim, hidden_dim // 8, heads=8, edge_dim=self.edge_dim))
                current_in_dim = hidden_dim # 8 heads * (hidden//8) = hidden_dim
            elif self.gnn_type == "gcn":
                self.convs.append(GCNConv(current_in_dim, hidden_dim))
                current_in_dim = hidden_dim

            self.norms.append(nn.LayerNorm(hidden_dim))

        # Only create JK scorer if requested, >1 layer, AND using attention
        if self.use_jk and self.gnn_layers > 1 and self.jk_mode == "attention":
            self.layer_scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )

        # --- DYNAMIC POOLING (MUST BE IN INIT) ---
        self.pooling_type = pooling_type
        if self.pooling_type == "gem":
            self.pool = GeMPooling(p=1.5)
        elif self.pooling_type == "mean":
            self.pool = global_mean_pool
        elif self.pooling_type == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        
        # Safety fallback
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ==========================================
        # EDGE STRATEGY ROUTING
        # ==========================================
        if self.edge_strategy == "attention":
            # 1. Project features into latent edge space
            proj_x = self.edge_proj(x)
            # 2. Generate edges based purely on feature similarity
            final_edge_index = knn_graph(proj_x, k=self.k_neighbors, batch=batch, loop=True)
            final_edge_attr = None

        elif self.edge_strategy == "hybrid":
            # 1. Generate kNN edges
            proj_x = self.edge_proj(x)
            knn_edges = knn_graph(proj_x, k=self.k_neighbors, batch=batch, loop=True)
            # 2. Concatenate them with the spatial edges
            final_edge_index = torch.cat([edge_index, knn_edges], dim=1)
            final_edge_attr = None

        else: # "spatial"
            # 1. Keep the exact edges generated by the SEEDS algorithm
            final_edge_index = edge_index
            final_edge_attr = edge_attr

        # --- SAFETY FALLBACK FOR GATv2 ---
        if self.edge_dim is not None and final_edge_attr is None:
            final_edge_attr = torch.zeros((final_edge_index.size(1), self.edge_dim), dtype=x.dtype, device=x.device)
        if not hasattr(self, '_printed_edge_stats'):
            print(f"\n[RUNTIME VERIFY] Strategy: {self.edge_strategy.upper()}")
            print(f"--> Nodes in batch: {x.size(0)}")
            print(f"--> Original (Spatial) Edges: {edge_index.size(1)}")
            print(f"--> Active (Final) Edges: {final_edge_index.size(1)}")
            self._printed_edge_stats = True
        # ==========================================
        # GRAPH AUGMENTATIONS
        # ==========================================
        if self.training:
            # 1. Modality Dropout 
            curr_x = self.apply_modality_dropout(x, self.drop_modality)
            
            # 2. Structural Dropout
            final_edge_index, _, _ = dropout_node(final_edge_index, p=self.drop_node, num_nodes=curr_x.size(0))
            final_edge_index, edge_mask = dropout_edge(final_edge_index, p=self.drop_edge, force_undirected=True)
            if final_edge_attr is not None:
                final_edge_attr = final_edge_attr[edge_mask]
        else:
            curr_x = x
        
        # --- DYNAMIC FORWARD PASS ---
        xs = []
        for i in range(self.gnn_layers):
            if self.gnn_type == "gatv2":
                curr_x = self.convs[i](curr_x, final_edge_index, edge_attr=final_edge_attr)
            else: # GCN baseline
                curr_x = self.convs[i](curr_x, final_edge_index)
                
            curr_x = F.elu(self.norms[i](curr_x))
            
            # Residual skip connections for deeper layers
            if i > 0: 
                curr_x = curr_x + xs[-1]
            xs.append(curr_x)

        # --- DYNAMIC JUMPING KNOWLEDGE ---
        if self.use_jk and self.gnn_layers > 1:
            x_stacked = torch.stack(xs, dim=1)
            
            if self.jk_mode == "attention":
                scores = self.layer_scorer(x_stacked)
                score_weights = F.softmin(scores, dim=1)
                x_dynamic = (x_stacked * score_weights).sum(dim=1)
            elif self.jk_mode == "mean":
                x_dynamic = x_stacked.mean(dim=1)
        else:
            x_dynamic = xs[-1]

        # --- DYNAMIC POOLING PASS ---
        pooled = self.pool(x_dynamic, batch)
        
        return pooled

    def apply_modality_dropout(self, x, p):
        """
        Randomly sets p percent of features of one modality (e.g. texture, cae, color) to zero per node.
        Dynamically calculates indices based on args.features.
        """
        if not self.training or p <= 0.0:
            return x

        x_dropped = x.clone()
        num_nodes = x.size(0) 

        feature_block_sizes = {
            'color': 3,
            'pos':   2,
            'hog':   self.num_hog_bins,
            'shape': 14,
            'lbp':   10,
            'texture': 1
        }
        
        feature_block_sizes['cae'] = self.cae_latent_dim
        extraction_order = ['color', 'pos', 'hog', 'cae', 'shape', 'lbp', 'texture']
        
        feature_blocks = {}
        curr_indx = 0
        
        for feat in extraction_order:
            if feat in self.features:
                size = feature_block_sizes[feat]
                feature_blocks[feat] = (curr_indx, curr_indx + size)
                curr_indx += size
                
        for name, (start, end) in feature_blocks.items():
            drop_mask = torch.rand(num_nodes, 1, device=x.device) < p
            x_dropped[:, start:end].masked_fill_(drop_mask, 0.0)

        return x_dropped
    

    
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
        
        # Load weights and set strict=False to ignore the missing classifier!
        model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
        model.eval()
        _WORKER_CAE_CACHE[weights_path] = model
        
    return _WORKER_CAE_CACHE[weights_path]

