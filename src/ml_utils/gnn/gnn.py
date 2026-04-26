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
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, knn_graph, SAGEConv
from PIL import Image
from helper import per_pixel_hog_bins
from gnn.cae import TextureEncoder
from torchvision import transforms
from skimage.measure import regionprops
import math
from torch_geometric.utils import dropout_edge, dropout_node, subgraph

def image_to_superpixel_graph(img, scale, sigma, min_size, hog_bins, mask=None , hog_signed=False, hog_l2norm=True, features=['color', 'pos', 'hog'], n_hops=1, cae_weights_path=None, cae_latent_dim=64, return_segments=False):   
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w, _ = img.shape

    #  felzenszwalb 
    segments = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    num_nodes = segments.max() + 1
    seg_flat = torch.tensor(segments, dtype=torch.long).view(-1)
    
    x_list = []

    # CAE SETUP 
    if 'cae' in features:
        if cae_weights_path is None:
            raise ValueError("Requested 'cae' features but no cae_weights_path provided!")
        cae_model = get_cae_model(cae_weights_path, cae_latent_dim)
        
        cae_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)), # MUST BE 64 TO MATCH CAE!
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ==========================================
    # --- GRAPH PRUNING (Mask Evaluation DEPRECATED DO NOT USE) ---
    # ==========================================
    if mask is not None:
        mask_flat = torch.tensor(mask, dtype=torch.float).view(-1)
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).float()
        
        node_mask_scores = torch.zeros(num_nodes, dtype=torch.float)
        node_mask_scores.scatter_add_(0, seg_flat, mask_flat)
        node_mask_scores = node_mask_scores / (node_counts + 1e-6)

        # Boolean array: True if it's the animal, False if it's background
        is_valid_node = node_mask_scores > 127.0 
    else:
        is_valid_node = torch.ones(num_nodes, dtype=torch.bool)

    # Safety net: If the augmentation cropped pure black and deleted everything
    if is_valid_node.sum() == 0:
        is_valid_node = torch.ones(num_nodes, dtype=torch.bool)


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
            # SKIP THE MATH IF IT IS BACKGROUND
            if not is_valid_node[sp]:
                # Just append dummy zeros to keep the list lengths intact
                if 'color' in features: colors.append([0, 0, 0])
                if 'pos' in features: positions.append([0, 0])
                if 'hog' in features: hogs.append(np.zeros(hog_bins, dtype=np.float32))
                if 'shape' in features: shape_features.append(np.zeros(14, dtype=np.float32))
                if 'cae' in features: cae_features.append(np.zeros(cae_latent_dim, dtype=np.float32))
                continue

            # Standard Extraction
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
                
            if 'cae' in features:
                ymin, ymax = coords[:, 0].min(), coords[:, 0].max()
                xmin, xmax = coords[:, 1].min(), coords[:, 1].max()
                crop_img = img[ymin:ymax+1, xmin:xmax+1]
                crop_tensor = cae_transform(crop_img).unsqueeze(0)
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
                node_shape_vec = [area, perimeter, aspect_ratio, circularity, solidity, extent, eccentricity] + log_hu
                shape_features.append(node_shape_vec)

        # Append extracted loop features to main list
        if 'color' in features: x_list.append(torch.tensor(np.array(colors), dtype=torch.float))
        if 'pos' in features: x_list.append(torch.tensor(np.array(positions), dtype=torch.float))
        if 'hog' in features: x_list.append(torch.tensor(np.array(hogs), dtype=torch.float))
        if 'cae' in features: x_list.append(torch.tensor(np.array(cae_features), dtype=torch.float))
        if 'shape' in features:
            x_shape = torch.tensor(np.array(shape_features), dtype=torch.float)
            x_shape = F.normalize(x_shape, p=2, dim=0) 
            x_list.append(x_shape)

    # Local Binary Patterns (LBP) Histogram 
    if 'lbp' in features:
        gray = (rgb2gray(img) * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray, P=8, R=1.0, method='uniform')
        lbp_flat = torch.tensor(lbp, dtype=torch.long).view(-1)
        lbp_one_hot = F.one_hot(lbp_flat, num_classes=10).float()
        x_lbp = torch.zeros((num_nodes, 10), dtype=torch.float)
        x_lbp.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, 10), lbp_one_hot)
        x_lbp = F.normalize(x_lbp, p=1, dim=1) 
        x_list.append(x_lbp)

    # Local Entropy 
    if 'texture' in features:
        gray_uint8 = (rgb2gray(img) * 255).astype(np.uint8)
        ent = entropy(gray_uint8, disk(3))
        ent_flat = torch.tensor(ent, dtype=torch.float).view(-1, 1)
        node_counts = torch.bincount(seg_flat, minlength=num_nodes).view(-1, 1).float()
        x_ent = torch.zeros((num_nodes, 1), dtype=torch.float)
        x_ent.scatter_add_(0, seg_flat.unsqueeze(1), ent_flat)
        x_ent = x_ent / (node_counts + 1e-6)
        x_list.append(x_ent)

    # Combine all selected features
    x = torch.cat(x_list, dim=1).to(torch.bfloat16)

    # APPLY PRUNING: Only keep the valid nodes!
    x_pruned = x[is_valid_node]
    old_to_new_ids = torch.full((num_nodes,), -1, dtype=torch.long)
    old_to_new_ids[is_valid_node] = torch.arange(x_pruned.size(0))

    #  Build Pruned Edges 
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

    # ---------------------------------------------------------
    # Build PyG Edge Index Safely (Handle 0-edge graphs)
    # ---------------------------------------------------------
    if len(edges) > 0:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # ---------------------------------------------------------
    # Calculate Local Edge Attributes (Distance and Angle)
    # ---------------------------------------------------------
    edge_attr = None
    if 'pos' in features:
        if edge_index.numel() > 0:
            # Dynamically find where the 'pos' features are stored in x_pruned
            feature_block_sizes = {'color': 3, 'pos': 2, 'hog': hog_bins, 'shape': 14, 'lbp': 10, 'texture': 1, 'cae': cae_latent_dim}
            extraction_order = ['color', 'pos', 'hog', 'cae', 'shape', 'lbp', 'texture']
            
            pos_start = 0
            for feat in extraction_order:
                if feat == 'pos':
                    break
                if feat in features:
                    pos_start += feature_block_sizes[feat]
                    
            # Extract positions and cast to float32 for math functions
            pos_tensor = x_pruned[:, pos_start : pos_start+2].to(torch.float32)
            row, col = edge_index
            
            # Calculate Relative Vector
            rel_pos = pos_tensor[row] - pos_tensor[col]
            
            # 1. Distance
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)
            # 2. Angle (Sine and Cosine)
            angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
            sin_angle = torch.sin(angle).unsqueeze(-1)
            cos_angle = torch.cos(angle).unsqueeze(-1)
            
            # Concatenate and cast back to bfloat16 to match your network
            edge_attr = torch.cat([dist, sin_angle, cos_angle], dim=-1).to(torch.bfloat16)
        else:
            # FIX: If there are no edges, supply an empty [0, 3] tensor instead of None!
            edge_attr = torch.empty((0, 3), dtype=torch.bfloat16)

    data = Data(x=x_pruned, edge_index=edge_index, edge_attr=edge_attr)
        
    if n_hops > 1:
        actual_nodes = data.x.size(0)
        adj = torch.zeros((actual_nodes, actual_nodes), dtype=torch.float)
        adj[data.edge_index[0], data.edge_index[1]] = 1.0
        adj.fill_diagonal_(1.0)
        adj_n = torch.matrix_power(adj, n_hops)
        adj_n.fill_diagonal_(0.0)
        new_edge_index = (adj_n > 0).nonzero(as_tuple=False).t().contiguous()
        data.edge_index = new_edge_index
        
    if return_segments:
        return data, segments
        
    return data

class BackgroundPruner(nn.Module):
    def __init__(self, in_dim, threshold = 0.1):
        super().__init__()
        self.threshold = threshold
        # light layer to gain information over node neighbourhood
        self.context_layer = SAGEConv(in_dim, in_dim//2)
        self.prune_scorer = nn.Sequential(nn.BatchNorm1d(in_dim//2),
                                          nn.ReLU(),
                                          nn.Linear(in_dim//2, 1),
                                          nn.Sigmoid())
        
    def forward(self, x, edge_index, batch = None):
        # create "context"
        x_context = self.context_layer(x,edge_index)

        # judge /score each node
        scores = self.prune_scorer(x_context).squeeze(-1)

        # gradient flow needs this
        x_weighted = x * scores.unsqueeze(-1)

        keep_mask = scores > self.threshold

        x_pruned = x_weighted[keep_mask]
        batch_pruned = batch[keep_mask] if batch is not None else None
        #sever deleted edges and relabel surviving nodes
        edge_index_pruned, _ = subgraph(
            subset=keep_mask, 
            edge_index=edge_index, 
            relabel_nodes=True, 
            num_nodes=x.size(0)
        )
        
        return x_pruned, edge_index_pruned, batch_pruned, scores
    
class GNNEncoder(nn.Module):
    def __init__(self, features, cae_latent_dim , num_hog_bins,in_dim=14, hidden_dim=512, out_dim=512, edge_strategy="spatial", k_neighbors=5):
        super().__init__()
        # The number of attention-based semantic edges to create per node
        self.k_neighbors = k_neighbors
        self.edge_strategy = edge_strategy
        self.features = features
        self.cae_latent_dim = cae_latent_dim
        self.num_hog_bins = num_hog_bins
        #self.pruner = BackgroundPruner(in_dim=in_dim)
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
        self.edge_dim = 3 if 'pos' in features else None
        
        # Pass edge_dim into the GATv2Conv layers
        self.conv1 = GATv2Conv(in_dim, hidden_dim//8, heads=8, edge_dim=self.edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, edge_dim=self.edge_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, edge_dim=self.edge_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        
        # Scores the 512-dim feature vectors to decide which layer is most useful
        self.layer_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Initialize the GeM Pooling layer here
        self.gem_pool = GeMPooling(p=1.5)

    def forward(self, data):
        x, spatial_edge_index, batch = data.x, data.edge_index, data.batch
        
        # Safely extract edge_attr if it exists
        edge_attr = getattr(data, 'edge_attr', None) 

        if self.training:
            x = self.apply_modality_dropout(x, p=0.2)
        
        if self.edge_strategy == "spatial":
            final_edge_index = spatial_edge_index
            final_edge_attr = edge_attr # Keep spatial edge attributes
            
        else:
            queries_keys = self.edge_proj(x)
            semantic_edge_index = knn_graph(x=queries_keys, k=self.k_neighbors, batch=batch, loop=False, cosine=True)

            if self.edge_strategy == "attention":
                final_edge_index = semantic_edge_index
                final_edge_attr = None # Attention edges don't have spatial attributes

            elif self.edge_strategy == "hybrid":
                final_edge_index = torch.cat([spatial_edge_index, semantic_edge_index], dim=1)
                final_edge_attr = None # Cannot mix spatial and non-spatial attributes

       # Graph augmentations
        if self.training:
            final_edge_index, _, _ = dropout_node(
                final_edge_index, 
                p=0.10, 
                num_nodes=x.size(0)
            )
            
            # Extract the edge_mask so we can drop the corresponding edge attributes!
            final_edge_index, edge_mask = dropout_edge(
                final_edge_index, 
                p=0.15, 
                force_undirected=True
            )
            # Sync edge attributes with the dropped edges
            if final_edge_attr is not None:
                final_edge_attr = final_edge_attr[edge_mask]
        
        # Layer 1 (Pass final_edge_attr)
        x1 = self.norm1(F.elu(self.conv1(x, final_edge_index, edge_attr=final_edge_attr)))
        
        # Layer 2 (with Residual Skip Connection)
        x2 = self.norm2(F.elu(self.conv2(x1, final_edge_index, edge_attr=final_edge_attr)))
        x2 = x2 + x1

        # Layer 3
        x3 = self.norm3(F.elu(self.conv3(x2, final_edge_index, edge_attr=final_edge_attr)))
        x3 = F.softplus(x3)

        # ==============================================================
        # RESTORED: Dynamic Jumping Knowledge (Layer Attention)
        # ==============================================================
        # reshape to [Num_Nodes, 3, 512]
        x_stacked = torch.stack([x1, x2, x3], dim=1)
        scores = self.layer_scorer(x_stacked)

        # convert to percentage
        score_weights = F.softmin(scores, dim=1)

        # Multiply and sum to get the final custom blend per node
        x_dynamic = (x_stacked * score_weights).sum(dim=1)

        # ==============================================================
        # RESTORED: Pooling and Return
        # ==============================================================
        pooled = self.gem_pool(x_dynamic, batch)
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

        # 1. Define the sizes of each block
        feature_block_sizes = {
            'color': 3,
            'pos':   2,
            'hog':   self.num_hog_bins,
            'shape': 14,
            'lbp':   10,
            'texture': 1
        }
        
        # Dynamically pull the CAE size  if it exists, otherwise default to 16
        feature_block_sizes['cae'] = self.cae_latent_dim

        # 2. strict order they are appended in image_to_superpixel_graph
        extraction_order = ['color', 'pos', 'hog', 'cae', 'shape', 'lbp', 'texture']
        
        feature_blocks = {}
        curr_indx = 0
        
        # 3. Build the dynamic index map based ONLY on what is active
        for feat in extraction_order:
            if feat in self.features:
                size = feature_block_sizes[feat]
                # Map the feature to its start and end indices
                feature_blocks[feat] = (curr_indx, curr_indx + size)
                curr_indx += size
                
        # 4. Apply the Dropout
        for name, (start, end) in feature_blocks.items():
            # True = Drop this modality for this node
            drop_mask = torch.rand(num_nodes, 1, device=x.device) < p
            
            # Fill the selected feature columns with 0.0 where the mask is True
            x_dropped[:, start:end].masked_fill_(drop_mask, 0.0)

        # Missing in previous code: you must return the modified tensor!
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

