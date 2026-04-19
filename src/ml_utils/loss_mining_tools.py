from torch.utils.data import Sampler
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class PKBatchSampler(Sampler):
    def __init__(self, labels, P=6, K=4, drop_last=True, alpha = 0.5):
        self.labels = np.asarray(labels)
        self.P = P
        self.K = K
        self.drop_last = drop_last
        # alpha = penalty factor for IDs with many occurancess
        self.alpha = alpha

        self.label_to_indices = {}
        for i, y in enumerate(self.labels):
            self.label_to_indices.setdefault(int(y), []).append(i)

        self.unique_labels = list(self.label_to_indices.keys())

        # --- WEIGHT CALCULATION ---
        # Count the number of images each identity has
        counts = np.array([len(self.label_to_indices[y]) for y in self.unique_labels])

        # Apply the penalty: 1 / (count^alpha)
        # use a small epsilon to prevent any theoretical division by zero
        weights = 1.0 / ((counts ** self.alpha) + 1e-8)

        # Normalize weights so they sum to 1.0 (creating a valid probability distribution)
        self.probabilities = weights / weights.sum()

    def __iter__(self):
        n_batches = len(self)
        for _ in range(n_batches):
            # Sample P identities using our calculated probability distribution
            if len(self.unique_labels) >= self.P:
                chosen = np.random.choice(
                    self.unique_labels, 
                    size=self.P, 
                    replace=False, 
                    p=self.probabilities
                )
            else:
                chosen = np.random.choice(
                    self.unique_labels, 
                    size=self.P, 
                    replace=True, 
                    p=self.probabilities
                )

            batch = []
            for y in chosen:
                idxs = self.label_to_indices[y]
                # sample K examples per identity (with replacement if needed)
                if len(idxs) >= self.K:
                    batch.extend(random.sample(idxs, self.K))
                else:
                    batch.extend(random.choices(idxs, k=self.K))

            yield batch
            
    def __len__(self):
        # rough epoch length; you can tune this
        n = len(self.labels)
        b = self.P * self.K
        return n // b if self.drop_last else (n + b - 1) // b

def pairwise_dist(x):
    # x: (B, D), normalized embeddings recommended
    # squared euclidean distances
    xx = (x * x).sum(dim=1, keepdim=True)
    dist = xx + xx.t() - 2.0 * (x @ x.t())
    return dist.clamp_min(0.0)

def batch_topk_triplet_loss(embeddings, labels, margin=1.0, k_pos=10, k_neg=10):
    """
    Functional implementation of Top-K Hard Triplet Loss.
    """
    # 1. Compute Pairwise Distance Matrix
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    # 2. Create Boolean Masks
    N = labels.size(0)
    is_same = labels.unsqueeze(0) == labels.unsqueeze(1)
    is_self = torch.eye(N, dtype=torch.bool, device=embeddings.device)

    pos_mask = is_same & ~is_self
    neg_mask = ~is_same

    # 3. POSITIVE MINING (Furthest)
    pos_dists = dist_mat.clone()
    pos_dists[~pos_mask] = -float('inf')

    actual_k_pos = max(1, min(k_pos, pos_mask.sum(dim=1).max().item()))
    top_pos_dists, _ = torch.topk(pos_dists, k=actual_k_pos, dim=1, largest=True)
    
    valid_pos = top_pos_dists > -1e5
    mean_hard_pos = (top_pos_dists * valid_pos).sum(dim=1) / valid_pos.sum(dim=1).clamp(min=1)

    # 4. NEGATIVE MINING (Closest)
    neg_dists = dist_mat.clone()
    neg_dists[~neg_mask] = float('inf')

    actual_k_neg = max(1, min(k_neg, neg_mask.sum(dim=1).max().item()))
    top_neg_dists, _ = torch.topk(neg_dists, k=actual_k_neg, dim=1, largest=False)
    
    valid_neg = top_neg_dists < 1e5
    mean_hard_neg = (top_neg_dists * valid_neg).sum(dim=1) / valid_neg.sum(dim=1).clamp(min=1)

    # 5. COMPUTE TRIPLET LOSS
    #losses = F.relu(mean_hard_pos - mean_hard_neg + margin)
    losses = F.softplus(mean_hard_pos - mean_hard_neg)
    return losses.mean()
    
def batch_hard_triplet_loss(emb, labels, margin=1.0):
    # emb: (B, D), labels: (B,)
    dist = pairwise_dist(emb)

    labels = labels.view(-1, 1)
    same = (labels == labels.t())
    diff = ~same

    # exclude self from positives
    eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    same = same & ~eye

    # hardest positive: max dist among same-ID
    pos_dist = dist.clone()
    pos_dist[~same] = -1.0
    hardest_pos, _ = pos_dist.max(dim=1)

    # hardest negative: min dist among different-ID
    neg_dist = dist.clone()
    neg_dist[~diff] = 1e9
    hardest_neg, _ = neg_dist.min(dim=1)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()

def batch_semi_hard_triplet_loss(emb, labels, margin=0.2):
    dist = pairwise_dist(emb)  # (B,B)

    labels = labels.view(-1, 1)
    same = (labels == labels.t())
    diff = ~same

    eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    same = same & ~eye

    # hardest positive (max among positives)
    pos = dist.clone()
    pos[~same] = -1.0
    hardest_pos, _ = pos.max(dim=1)  # (B,)

    # semi-hard negatives: d(ap) < d(an) < d(ap)+margin
    ap = hardest_pos.view(-1, 1)  # (B,1)
    semi = diff & (dist > ap) & (dist < (ap + margin))

    # pick closest semi-hard negative if exists, else closest negative
    neg = dist.clone()
    neg[~diff] = 1e9
    closest_neg, _ = neg.min(dim=1)

    semi_neg = dist.clone()
    semi_neg[~semi] = 1e9
    closest_semi, _ = semi_neg.min(dim=1)

    use_semi = (closest_semi < 1e8)
    chosen_neg = torch.where(use_semi, closest_semi, closest_neg)

    return F.relu(hardest_pos - chosen_neg + margin).mean()

def batch_topk_semi_hard_triplet_loss(emb, labels, margin=0.2, k_neg=3):
    dist = pairwise_dist(emb)

    labels = labels.view(-1, 1)
    same = (labels == labels.t())
    diff = ~same

    eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
    same = same & ~eye

    # 1. Find the hardest positive for each anchor
    pos = dist.clone()
    pos[~same] = -1.0
    hardest_pos, _ = pos.max(dim=1)
    ap = hardest_pos.view(-1, 1)

    # 2. Identify the Semi-Hard Zone: d(ap) < d(an) < d(ap)+margin
    semi = diff & (dist > ap) & (dist < (ap + margin))

    # 3. Mask out everything that isn't semi-hard
    semi_neg = dist.clone()
    semi_neg[~semi] = 1e9

    # 4. Grab the Top K closest semi-hard negatives
    # (Using largest=False because smaller distance = harder)
    closest_k_semi, _ = semi_neg.topk(k_neg, dim=1, largest=False)
    
    # 5. Create a boolean mask of which ones are actually valid 
    # (in case there were fewer than K semi-hards available)
    valid_semi = closest_k_semi < 1e8

    # 6. Fallback: Identify the absolute closest negative overall
    neg = dist.clone()
    neg[~diff] = 1e9
    closest_neg, _ = neg.min(dim=1)

    losses = []
    
    # 7. Calculate the loss per anchor
    for i in range(dist.size(0)):
        # Extract only the valid semi-hard distances for this anchor
        valid_dists = closest_k_semi[i][valid_semi[i]]
        
        if len(valid_dists) > 0:
            # Average the loss over the 1 to K available semi-hard negatives
            loss_i = F.relu(hardest_pos[i] - valid_dists + margin).mean()
        else:
            # Fallback: If no semi-hards exist, just use the single hardest negative
            loss_i = F.relu(hardest_pos[i] - closest_neg[i] + margin)
            
        losses.append(loss_i)

    return torch.stack(losses).mean()

def batch_compactness_loss(embeddings, labels):
    """
    Calculates the variance of embeddings from their class centers within a single batch.
    Penalizes embeddings that stray too far from their batch-wise center.
    """
    loss = 0.0
    unique_labels = torch.unique(labels)
    valid_classes = 0
    
    for label in unique_labels:
        # 1. Find all embeddings for this specific animal in the batch
        mask = (labels == label)
        class_embs = embeddings[mask]
        
        # 2. We only calculate variance if there are at least 2 images of this animal
        if len(class_embs) > 1:
            # Calculate the "center of mass" for this animal in this batch
            center = class_embs.mean(dim=0)
            
            # Penalize the Mean Squared Error (distance) between each image and the center
            loss += F.mse_loss(class_embs, center.expand_as(class_embs))
            valid_classes += 1
            
    # Average the loss across the number of valid identities in the batch
    if valid_classes > 0:
        return loss / valid_classes
    
    # Fallback if no classes had >1 image (shouldn't happen with your PK sampler)
    return torch.tensor(0.0, device=embeddings.device, requires_grad=True)