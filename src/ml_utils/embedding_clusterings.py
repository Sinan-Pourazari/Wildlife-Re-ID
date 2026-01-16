import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def cosine_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    query = l2_normalize(query)
    candidates = l2_normalize(candidates)
    return 1.0 - (candidates @ query)


# ------------------------------------------------------------
# Identity memory (Option 2)
# ------------------------------------------------------------

class IdentityMemory:
    """
    Simple open-set identity memory (image-only, exemplar-based).
    """

    def __init__(self, threshold: float = 0.35, max_exemplars_per_identity: int = 10):
        self.threshold = threshold
        self.max_exemplars_per_identity = max_exemplars_per_identity

        # identity_id -> list of embeddings
        self.memory: Dict[int, List[torch.Tensor]] = {}

        # counter for new identities
        self.next_identity_id: int = 0

    # --------------------------------------------------------

    def create_identity(self, embedding: torch.Tensor) -> int:
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.memory[identity_id] = [embedding.detach().cpu()]
        return identity_id

    def add_exemplar(self, identity_id: int, embedding: torch.Tensor) -> None:
        self.memory[identity_id].append(embedding.detach().cpu())

        # FIFO cap
        if len(self.memory[identity_id]) > self.max_exemplars_per_identity:
            self.memory[identity_id] = self.memory[identity_id][-self.max_exemplars_per_identity:]

    # --------------------------------------------------------

    def distance_to_identity(self, query_embedding: torch.Tensor, identity_id: int) -> float:
        exemplars = self.memory[identity_id]
        exemplar_tensor = torch.stack(exemplars, dim=0)
        distances = cosine_distance(query_embedding.view(-1), exemplar_tensor)
        return float(distances.min().item())

    # --------------------------------------------------------

    @torch.no_grad()
    def match(self, query_embedding: torch.Tensor) -> Tuple[Optional[int], float]:
        if len(self.memory) == 0:
            return None, float("inf")

        best_id = None
        best_dist = float("inf")

        for identity_id in self.memory.keys():
            d = self.distance_to_identity(query_embedding, identity_id)
            if d < best_dist:
                best_dist = d
                best_id = identity_id

        if best_dist < self.threshold:
            return best_id, best_dist
        return None, best_dist

    # --------------------------------------------------------

    @torch.no_grad()
    def upsert(self, query_embedding: torch.Tensor) -> Tuple[int, float, bool]:
        matched_id, best_dist = self.match(query_embedding)

        if matched_id is None:
            new_id = self.create_identity(query_embedding)
            return new_id, best_dist, True
        else:
            self.add_exemplar(matched_id, query_embedding)
            return matched_id, best_dist, False
