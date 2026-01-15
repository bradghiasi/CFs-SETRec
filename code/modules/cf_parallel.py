import torch
import torch.nn as nn
import torch.nn.functional as F

class CFBranch(nn.Module):
    def __init__(self, cf_in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cf_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        nn.init.xavier_uniform_(self.proj[0].weight); nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[2].weight); nn.init.zeros_(self.proj[2].bias)

    def forward(self, cf_vec):
        return self.proj(cf_vec)

class BranchFuse(nn.Module):
    def __init__(self, mode: str, d_model: int, n_branches: int):
        super().__init__()
        self.mode = mode
        self.gate = nn.Linear(d_model, n_branches) if mode == 'learned' else None

    def forward(self, logits_per_branch, context=None):
        if self.mode == 'mean':
            return torch.stack(logits_per_branch, 0).mean(0)
        assert context is not None, "context is required for learned fusion"
        w = torch.softmax(self.gate(context), dim=-1)      # (B, K)
        L = torch.stack(logits_per_branch, -1)             # (B, n_items, K)
        return (L * w.unsqueeze(1)).sum(-1)                # (B, n_items)

class CFParallel(nn.Module):
    def __init__(self, cf_table: torch.Tensor, d_model: int, n_branches: int, fuse_mode: str):
        super().__init__()
        self.register_buffer("cf_table", cf_table, persistent=False)
        self.branches = nn.ModuleList([CFBranch(cf_table.size(1), d_model) for _ in range(n_branches)])
        self.fuser = BranchFuse(fuse_mode, d_model, n_branches)
        self.n_branches = n_branches

    def lookup(self, item_ids: torch.LongTensor):
        return self.cf_table[item_ids]  # (B, cf_in_dim)

    def tokens(self, item_ids: torch.LongTensor):
        base = self.lookup(item_ids)
        outs = [b(base) for b in self.branches]   # list of (B, d_model)
        return torch.stack(outs, 1)               # (B, K, d_model)
