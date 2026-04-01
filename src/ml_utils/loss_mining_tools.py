from torch.utils.data import Sampler
import random
import numpy as np
import torch
import torch.nn.functional as F

class PKBatchSampler(Sampler):
    def __init__(self, labels, P=8, K=4, drop_last=True):
        self.labels = np.asarray(labels)
        self.P = P
        self.K = K
        self.drop_last = drop_last

        self.label_to_indices = {}
        for i, y in enumerate(self.labels):
            self.label_to_indices.setdefault(int(y), []).append(i)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        n_batches = len(self)

        for _ in range(n_batches):
            # sample P identities
            if len(self.unique_labels) >= self.P:
                chosen = random.sample(self.unique_labels, self.P)
            else:
                chosen = random.choices(self.unique_labels, k=self.P)

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
