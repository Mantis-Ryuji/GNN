from __future__ import annotations

import sys
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import TrainConfig
from model import GINERegressor
from data import smiles_to_mol, mol_to_graph, collate_graphs


# ---------------------------------------------------------
# Inference Dataset（y はダミー 0.0 のまま）
# ---------------------------------------------------------
class InferenceDataset(Dataset):
    """
    事前に作っておいたグラフ（x, edge_index, edge_attr）から、
    collate_graphs が期待する形式の dict を返す Dataset。
    y はダミー（0.0）を入れる（collate_graphs の都合）。
    """
    def __init__(self, graphs: List[Dict[str, Any]]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = self.graphs[idx]
        return {
            "x": g["x"],                   # (N, in_dim)
            "edge_index": g["edge_index"], # (2, E)
            "edge_attr": g["edge_attr"],   # (E, edge_dim)
            "y": torch.tensor([0.0]),      # ダミー
        }


# ---------------------------------------------------------
# モデルロード：TrainConfig.device を使用
# ---------------------------------------------------------
def load_model(cfg: TrainConfig, in_dim: int, edge_dim: int):
    """
    src/runs/best.pt を読み込み、cfg.device に応じてモデルを構築。
    戻り値: (model, device)
    """
    device = torch.device(cfg.device)

    model = GINERegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    ckpt = torch.load("src/runs/best.pt", map_location=device)

    # checkpoint 形式に合わせて安全にロード
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


# ---------------------------------------------------------
# メイン処理：stdin → SMILES → グラフ → 推論 → print 出力
# ---------------------------------------------------------
def main():
    cfg = TrainConfig()

    # ------------------------------------------
    # stdin から CSV を読み込み
    # ------------------------------------------
    input_data = []
    for line in sys.stdin:
        input_data.append(line.strip().split(","))

    if len(input_data) < 2:
        raise ValueError("入力が空です。")

    df = pd.DataFrame(data=input_data[1:], columns=input_data[0])

    # 必須カラムチェック
    smiles_col = cfg.smiles_col
    if smiles_col not in df.columns:
        raise ValueError(f"入力CSVに {smiles_col} 列がありません。")

    # ------------------------------------------
    # SMILES → Mol → Graph へ変換
    # invalid SMILES は即エラー
    # ------------------------------------------
    graphs: List[Dict[str, Any]] = []
    for smi in df[smiles_col].tolist():
        mol = smiles_to_mol(smi)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smi}")

        # ★ mol_to_graph は (x, edge_index, edge_attr) を返す想定
        x, edge_index, edge_attr = mol_to_graph(mol)
        graphs.append(
            {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }
        )

    # shape 情報（推論時に必要）
    in_dim = graphs[0]["x"].shape[-1]
    edge_dim = graphs[0]["edge_attr"].shape[-1]

    # ------------------------------------------
    # モデルロード（device は cfg.device に従う）
    # ------------------------------------------
    model, device = load_model(cfg, in_dim=in_dim, edge_dim=edge_dim)

    # ------------------------------------------
    # DataLoader — yはダミーなので捨てる
    # ------------------------------------------
    dataset = InferenceDataset(graphs)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # 標準化の mean/std を読み込み
    scaler = torch.load("src/runs/target_scaler.pt", map_location="cpu")
    y_mean = float(scaler["mean"])
    y_std = float(scaler["std"])

    preds_z_list = []

    # ------------------------------------------
    # 推論ループ（model/device 使用）
    # ------------------------------------------
    for x, adj, edge_attr, mask, _ in loader:  # yは捨てる
        x = x.to(device)
        adj = adj.to(device)
        edge_attr = edge_attr.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            pred_z = model(x, adj, edge_attr, mask)  # 標準化(z)スケールの予測

        preds_z_list.append(pred_z.cpu())

    preds_z = torch.cat(preds_z_list, dim=0).squeeze(-1)  # (N,)

    # 逆transform: y = z * std + mean
    preds_orig = preds_z * y_std + y_mean
    preds_orig = preds_orig.numpy().tolist()

    # ------------------------------------------
    # 出力：1 行に 1 つずつ print
    # ------------------------------------------
    for val in preds_orig:
        print(val)


# ---------------------------------------------------------
if __name__ == "__main__":
    main()