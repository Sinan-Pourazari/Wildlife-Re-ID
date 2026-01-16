import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb



def visualize_superpixels(image, segments, figsize=(10, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.canvas.manager.set_window_title('Superpixel_Prototype_Visualizer')
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    boundaries = mark_boundaries(image, segments, color=(1, 0, 0))
    axes[1].imshow(boundaries)
    axes[1].set_title("Superpixel Boundaries")
    axes[1].axis("off")

    # FIXED: pass the image
    sp_color = label2rgb(segments, image=image, bg_label=-1, kind='avg')
    axes[2].imshow(sp_color)
    axes[2].set_title("Superpixels (Colored by Avg)")
    axes[2].axis("off")
    



    plt.tight_layout()
    plt.show()




def visualize_superpixels_with_graph(
    image,
    segments,
    figsize=(14, 6),
    node_size=40,
    edge_alpha=0.35,
    save_path=None,          # ← NEW (optional)
    dpi=300,                 # ← NEW
):
    """
    Visualizes:
    1) Original image
    2) Superpixel boundaries
    3) Superpixels colored by average color
    4) GNN graph with nodes centered in superpixels

    If save_path is provided, the figure is saved with a transparent background.
    """

    h, w = segments.shape
    num_nodes = segments.max() + 1

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.canvas.manager.set_window_title("Superpixel + GNN Visualizer")

    # ---------- MAKE FIGURE TRANSPARENT ----------
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)
    # --------------------------------------------

    # --------------------------------------------------
    # 1) Original image
    # --------------------------------------------------
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # --------------------------------------------------
    # 2) Superpixel boundaries
    # --------------------------------------------------
    boundaries = mark_boundaries(image, segments, color=(1, 0, 0))
    axes[1].imshow(boundaries)
    axes[1].set_title("Superpixel Boundaries")
    axes[1].axis("off")

    # --------------------------------------------------
    # 3) Superpixels (avg color)
    # --------------------------------------------------
    sp_color = label2rgb(segments, image=image, bg_label=-1, kind="avg")
    axes[2].imshow(sp_color)
    axes[2].set_title("Superpixels (Avg Color)")
    axes[2].axis("off")

    # --------------------------------------------------
    # 4) GNN graph visualization
    # --------------------------------------------------
    graph_ax = axes[3]

    graph_ax.imshow(boundaries, alpha=1.0)
    graph_ax.set_title("GNN Graph (Nodes centered in Superpixels)")
    graph_ax.axis("off")

    # ---- compute centroids ----
    centroids = np.zeros((num_nodes, 2))
    for sp in range(num_nodes):
        ys, xs = np.where(segments == sp)
        centroids[sp] = [xs.mean(), ys.mean()]

    # ---- compute adjacency ----
    edges = set()
    for y in range(h - 1):
        for x in range(w - 1):
            a = segments[y, x]
            b = segments[y, x + 1]
            c = segments[y + 1, x]

            if a != b:
                edges.add((a, b))
                edges.add((b, a))
            if a != c:
                edges.add((a, c))
                edges.add((c, a))

    # ---- draw edges ----
    for a, b in edges:
        x1, y1 = centroids[a]
        x2, y2 = centroids[b]
        graph_ax.plot(
            [x1, x2],
            [y1, y2],
            color="black",
            linewidth=0.99,
            alpha=edge_alpha,
            zorder=1,
        )

    # ---- draw nodes ----
    graph_ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=node_size,
        c="yellow",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    plt.tight_layout()

    # ---------- SAVE WITH TRANSPARENCY ----------
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    # --------------------------------------------

    plt.show()

    import numpy as np

def per_pixel_hog_bins(img, n_bins=9, signed=False):
    """
    Returns:
      bin_idx: (H, W) int in [0, n_bins-1]
      mag:     (H, W) float gradient magnitude
    """
    # Use L channel (stable) or grayscale; here: simple luminance from RGB
    # If img is uint8 RGB: convert to float32
    img_f = img.astype(np.float32)
    gray = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]

    # Gradients (simple finite differences)
    gy, gx = np.gradient(gray)  # gy: d/dy, gx: d/dx

    mag = np.sqrt(gx * gx + gy * gy)

    ang = np.degrees(np.arctan2(gy, gx))  # [-180, 180]
    if signed:
        # Map to [0, 360)
        ang = (ang + 360.0) % 360.0
        angle_range = 360.0
    else:
        # Map to [0, 180)
        ang = (ang + 180.0) % 180.0
        angle_range = 180.0

    bin_width = angle_range / n_bins
    bin_idx = np.floor(ang / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    return bin_idx, mag
