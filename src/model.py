from __future__ import annotations

import torch
import torch.nn as nn


class GINELayer(nn.Module):
    """
    GINE レイヤ:
      - LayerNorm + 残差接続 (PreNorm)
      - メッセージ: h_j + e_ij を H 次元のまま MLP に通す
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # PreNorm
        self.norm = nn.LayerNorm(hidden_dim)

        # メッセージ MLP: (H → H → H)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

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
        assert H == self.hidden_dim

        # PreNorm
        h_norm = self.norm(x)  # (B,N,H)

        # h_j を (B, N, N, H) にブロードキャスト（j: 2軸）
        h_j = h_norm.unsqueeze(1).expand(-1, N, N, -1)  # (B,N,N,H)

        # メッセージ: h_j + e_ij （H 次元のまま）
        m_in = h_j + edge_emb                 # (B,N,N,H)
        m_ij = self.msg_mlp(m_in)             # (B,N,N,H)

        # 非隣接は 0 にする
        m_ij = m_ij * adj.unsqueeze(-1)       # (B,N,N,H)

        # ∑_j m_ij
        msg = m_ij.sum(dim=2)                 # (B,N,H)

        # Δh をそのまま残差に加える
        delta = self.dropout(msg)             # (B,N,H)
        out = x + delta                       # (B,N,H)

        # padding ノードは再度 0
        out = out * mask.unsqueeze(-1)
        return out


class AttentionReadout(nn.Module):
    """
    Graph-level attention pooling:
      α_i = softmax(a^T tanh(W h_i))
      g   = Σ_i α_i h_i
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask):
        """
        h:    (B, N, H)
        mask: (B, N)
        """
        # (B,N,H) → (B,N,H)
        h_proj = torch.tanh(self.proj(h))

        # スカラースコア (B,N)
        scores = self.attn(h_proj).squeeze(-1)

        # マスク: padding ノードには -inf を足す
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax over ノード
        alpha = torch.softmax(scores, dim=1)  # (B,N)

        # 数値安定用: 全部 -inf の場合には NaN になるので、そのときは一様にする
        if torch.isnan(alpha).any():
            alpha = torch.where(
                mask > 0,
                torch.ones_like(alpha)
                / mask.sum(dim=1, keepdim=True).clamp(min=1.0),
                torch.zeros_like(alpha),
            )

        # attention 重み付き和
        g = (h * alpha.unsqueeze(-1)).sum(dim=1)  # (B,H)
        return g


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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 入力埋め込み
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # 軽量 GINE レイヤスタック
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        # readout: attention pooling
        self.readout = AttentionReadout(hidden_dim)

        # 最終回帰ヘッド
        self.pred_head = nn.Linear(hidden_dim, 1)

        # 出力側にも軽く dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, edge_attr, mask):
        """
        x:        (B,N,F_in)
        adj:      (B,N,N)
        edge_attr:(B,N,N,F_e)
        mask:     (B,N)
        """
        # 入力埋め込み
        h = self.node_proj(x)                   # (B,N,H)
        edge_emb = self.edge_proj(edge_attr)    # (B,N,N,H)

        # GINE レイヤを適用
        for layer in self.layers:
            h = layer(h, adj, edge_emb, mask)

        # Attention readout
        g = self.readout(h, mask)               # (B,H)
        g = self.dropout(g)

        # 回帰出力
        out = self.pred_head(g).squeeze(-1)     # (B,)
        return out