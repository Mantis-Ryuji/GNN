from __future__ import annotations

import sys
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import TrainConfig
from model import GINERegressor
from data import smiles_to_mol, mol_to_graph, collate_graphs


class InferenceDataset(Dataset):
    """
    事前に作っておいたグラフ（x, edge_index, edge_attr）から、
    collate_graphs が期待する形式の dict を返す Dataset。
    y はダミー（0.0）を入れる。
    """
    def __init__(self, graphs: List[Dict[str, Any]]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = self.graphs[idx]
        return {
            "x": g["x"],                  # (N, F_in)
            "edge_index": g["edge_index"],  # (2, E)
            "edge_attr": g["edge_attr"],    # (E, F_e)
            "y": torch.tensor(0.0, dtype=torch.float32),
        }


def build_graphs_or_error(smiles_list: List[str]) -> List[Dict[str, Any]]:
    """
    SMILES から (x, edge_index, edge_attr) を構築する。
    1つでも invalid SMILES があれば即エラーを投げる。
    """
    graphs: List[Dict[str, Any]] = []
    for smi in smiles_list:
        mol = smiles_to_mol(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES detected: {smi}")
        x, edge_index, edge_attr = mol_to_graph(mol)  # (N,F), (2,E), (E,Fe)
        graphs.append(
            {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }
        )
    return graphs


def load_model(cfg: TrainConfig, in_dim: int, edge_dim: int) -> GINERegressor:
    """
    CPU 上に GINERegressor を構築し、src/runs/best.pt の state_dict をロードする。
    best.pt は train.py 側で {"epoch":..., "model": state_dict, "best_val_rmse":...}
    という形式で保存されている前提。
    """
    device = torch.device("cpu")

    model = GINERegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    ckpt = torch.load("src/runs/best.pt", map_location=device)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_target_scaler(path: str = "src/runs/target_scaler.pt") -> tuple[float, float]:
    """
    train.py で保存した target の mean / std をロードする。
    """
    obj = torch.load(path, map_location="cpu")
    mean = float(obj["mean"])
    std = float(obj["std"])
    if std <= 0.0:
        raise ValueError(f"Loaded target std is non-positive (std={std}).")
    return mean, std


def main():
    # ===== 1. stdin → DataFrame =====
    input_data = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        input_data.append(line.split(","))

    if len(input_data) == 0:
        return

    df = pd.DataFrame(data=input_data[1:], columns=input_data[0])

    cfg = TrainConfig()
    smiles_col = cfg.smiles_col

    if smiles_col not in df.columns:
        raise ValueError(f"入力に '{smiles_col}' 列がありません。")

    smiles_list = df[smiles_col].astype(str).tolist()

    # ===== 2. SMILES → グラフ (invalid なら即エラー) =====
    graphs = build_graphs_or_error(smiles_list)
    in_dim = graphs[0]["x"].shape[1]
    edge_dim = graphs[0]["edge_attr"].shape[1]

    # ===== 3. モデルロード (CPU) =====
    torch.set_grad_enabled(False)
    model = load_model(cfg, in_dim=in_dim, edge_dim=edge_dim)
    device = torch.device("cpu")

    # ===== 4. target スケーラのロード (mean, std) =====
    y_mean, y_std = load_target_scaler()

    # ===== 5. DataLoader でバッチ推論 =====
    dataset = InferenceDataset(graphs)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_graphs,  # train と同じ collate を再利用
    )

    preds: List[float] = []

    for x, adj, edge_attr, mask, _ in loader:  # y はダミーなので無視
        x = x.to(device)
        adj = adj.to(device)
        edge_attr = edge_attr.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            y_hat = model(x, adj, edge_attr, mask)  # shape: (B,)

        preds.extend(y_hat.cpu().tolist())

    # ===== 6. 標準化空間 → 元の λmax スケールに逆変換して順番に print =====
    for p in preds:
        p_raw = p * y_std + y_mean
        print(float(p_raw))


if __name__ == "__main__":
    main()

# cat sample/sample.in.csv | python src/main.py