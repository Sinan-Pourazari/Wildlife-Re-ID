import torch
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.io import imread
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from PIL import Image
from helper import visualize_superpixels, visualize_superpixels_with_graph, per_pixel_hog_bins

def image_to_superpixel_graph(img, n_segments=300, hog_bins=9, hog_signed=False, hog_l2norm=True):
    #print(type(img))
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w, _ = img.shape

    # Superpixels
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=0)
    #helper.visualize_superpixels(img,segments)
    #helper.visualize_superpixels_with_graph(img, segments)
    # Convert to LAB for better color stability
    img_lab = rgb2lab(img)

    num_nodes = segments.max() + 1

    # new global HOG precompute (per-pixel bins + magnitudes)
    pix_bin_idx, pix_mag = per_pixel_hog_bins(img, n_bins=hog_bins, signed=hog_signed)

    # Node features: [mean_L, mean_A, mean_B, centroid_x, centroid_y]
    features = []
    positions = []

    for sp in range(num_nodes):
        mask = (segments == sp)
        coords = np.column_stack(np.nonzero(mask))
        vals = img_lab[mask]

        L, A, B = vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()
        y, x = coords[:, 0].mean(), coords[:, 1].mean()

        # superpixel-local histogram over GLOBAL bins (weighted by magnitude)
        sp_bins = pix_bin_idx[mask].ravel()
        sp_w = pix_mag[mask].ravel()
        # ensure bin indices are always in [0, hog_bins-1]
        sp_bins = np.clip(sp_bins, 0, hog_bins - 1)

        hog_hist = np.bincount(sp_bins, weights=sp_w, minlength=hog_bins).astype(np.float32)

        # enforce exact length 
        hog_hist = hog_hist[:hog_bins]

        # optional per-node histogram normalization
        if hog_l2norm:
            hog_hist /= (np.linalg.norm(hog_hist, ord=2) + 1e-6)

        # append HOG histogram bins to node feature vector
        node_feat = np.concatenate([[L, A, B, x / w, y / h], hog_hist], axis=0)
        features.append(node_feat)

        #features.append([L, A, B, x / w, y / h])
        positions.append([x, y])

    #stack because features are now numpy vectors
    x = torch.tensor(np.stack(features, axis=0), dtype=torch.float)

    # Build edges: adjacency from superpixel boundaries
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

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=512, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.output_dim = out_dim

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Graph-level embedding via global pooling
        batch = torch.zeros( x.size(0), dtype=torch.long, device=x.device)

        x = global_mean_pool(x, batch=batch)

        return self.lin(x)
"""
if __name__ == "__main__":
    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    img = imread(img_path)

    # --------------------------------------------------
    # Build superpixel graph
    # --------------------------------------------------
    graph = image_to_superpixel_graph(img, n_segments=150)

    # --------------------------------------------------
    # Initialize model
    # --------------------------------------------------
    model = GNNEncoder()
    model.eval()  # important for inference

    # --------------------------------------------------
    # Forward pass (single sample)
    # --------------------------------------------------
    with torch.no_grad():
        embedding = model(graph)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    print("Embedding shape:", embedding.shape)
    print("Embedding:", embedding)
    """