from __future__ import annotations

import torch
import torch.nn as nn


class GINELayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x, adj, edge_emb, mask):
        """
        x:        (B, N, H)
        adj:      (B, N, N)   0/1
        edge_emb: (B, N, N, H)
        mask:     (B, N)
        """
        # padding ノードをゼロに
        x = x * mask.unsqueeze(-1)  # (B,N,H)

        B, N, H = x.shape

        # 近傍メッセージ: ReLU(h_j + e_ij)
        # x_j: (B,1,N,H) → (B,N,N,H) （j が 2軸目）
        x_j = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B,N,N,H)

        msg_ij = torch.relu(x_j + edge_emb)  # (B,N,N,H)
        msg_ij = msg_ij * adj.unsqueeze(-1)  # 非隣接は0

        # ∑_j
        msg = msg_ij.sum(dim=2)  # (B,N,H)

        out = (1.0 + self.eps) * x + msg
        out = out * mask.unsqueeze(-1)

        out = self.mlp(out)  # (B,N,H)
        return out


class GINERegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj, edge_attr, mask):
        """
        x:        (B,N,F_in)
        adj:      (B,N,N)
        edge_attr:(B,N,N,F_e)
        mask:     (B,N)
        """
        # 入力埋め込み
        h = self.node_proj(x)  # (B,N,H)
        edge_emb = self.edge_proj(edge_attr)  # (B,N,N,H)

        for layer in self.layers:
            h = layer(h, adj, edge_emb, mask)
            h = self.dropout(h)

        # グラフプーリング (masked mean)
        mask_ = mask.unsqueeze(-1)  # (B,N,1)
        h = h * mask_
        sum_h = h.sum(dim=1)                 # (B,H)
        count = mask_.sum(dim=1).clamp(min=1.0)  # (B,1)
        g = sum_h / count                    # (B,H)

        out = self.readout(g).squeeze(-1)    # (B,)
        return out
