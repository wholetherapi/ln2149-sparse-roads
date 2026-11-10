import torch
import torch.nn as nn

class EventCuedSparseAttention(nn.Module):
    """Sparse attention whose token budget is set by an event cue."""

    def __init__(self, dim: int, heads: int, top_k: int = 32):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cue = nn.Linear(dim, 1)
        self.top_k = top_k

    def forward(self, x):
        scores = self.cue(x).squeeze(-1)
        keep = torch.topk(scores, k=min(self.top_k, x.shape[1]), dim=1).indices
        out, _ = self.attn(x, x, x)
        return out.gather(1, keep.unsqueeze(-1).expand(-1, -1, out.shape[-1]))
