from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from rdkit import Chem

# 利用する原子種
ATOM_LIST = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "B", "Si", "Se"]

# ハイブリダイゼーション種
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# ボンドタイプ
BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
EDGE_FEAT_DIM = len(BOND_TYPE_LIST) + 2  # bond-type one-hot + conjugated + ring


# -----------------------------
# SMILES / Mol → グラフ変換
# -----------------------------

def smiles_to_mol(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def atom_to_feat(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """原子 → ノード特徴ベクトル"""
    symbol = atom.GetSymbol()
    one_hot = torch.zeros(len(ATOM_LIST), dtype=torch.float32)
    idx = ATOM_LIST.index(symbol) if symbol in ATOM_LIST else -1
    if idx >= 0:
        one_hot[idx] = 1.0

    # 既存の4スカラー特徴
    base_scalars = torch.tensor(
        [
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            atom.GetTotalDegree(),
            float(atom.IsInRing()),
        ],
        dtype=torch.float32,
    )

    # ハイブリダイゼーション one-hot
    hyb = atom.GetHybridization()
    hyb_one_hot = torch.zeros(len(HYB_LIST), dtype=torch.float32)
    if hyb in HYB_LIST:
        hyb_one_hot[HYB_LIST.index(hyb)] = 1.0

    # 価数 & 原子番号（スカラー）
    valence = float(atom.GetTotalValence())
    atomic_num = float(atom.GetAtomicNum())

    extra_scalars = torch.tensor([valence, atomic_num], dtype=torch.float32)

    feats = torch.cat(
        [
            one_hot,          # |ATOM_LIST|
            base_scalars,     # 4
            hyb_one_hot,      # |HYB_LIST|
            extra_scalars,    # 2
        ],
        dim=0,
    )
    return feats  # (F,)


def bond_to_feat(bond: Chem.rdchem.Bond) -> torch.Tensor:
    """ボンド → エッジ特徴ベクトル"""
    bt = bond.GetBondType()
    one_hot = torch.zeros(len(BOND_TYPE_LIST), dtype=torch.float32)
    if bt in BOND_TYPE_LIST:
        one_hot[BOND_TYPE_LIST.index(bt)] = 1.0

    conj = float(bond.GetIsConjugated())
    ring = float(bond.IsInRing())

    feats = torch.cat(
        [
            one_hot,                   # bond type
            torch.tensor([conj, ring], dtype=torch.float32),
        ],
        dim=0,
    )
    return feats  # (EDGE_FEAT_DIM,)


def mol_to_graph(
    mol: Chem.Mol,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    RDKit Mol -> (node_feats, edge_index, edge_attr)

    node_feats: (N, F)
    edge_index: (2, E)  無向グラフなので双方向を格納
    edge_attr:  (E, EDGE_FEAT_DIM)
    """
    num_atoms = mol.GetNumAtoms()
    node_feats = torch.stack(
        [atom_to_feat(mol.GetAtomWithIdx(i)) for i in range(num_atoms)], dim=0
    )  # (N, F)

    src: List[int] = []
    dst: List[int] = []
    edge_attr_list: List[torch.Tensor] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_to_feat(bond)
        # 無向 → 双方向
        src += [i, j]
        dst += [j, i]
        edge_attr_list += [feat, feat]

    if len(src) == 0:
        # 単原子分子など edge が無い場合
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)  # (2, E)
        edge_attr = torch.stack(edge_attr_list, dim=0)           # (E, EDGE_FEAT_DIM)

    return node_feats, edge_index, edge_attr


# -----------------------------
# Dataset & collate
# -----------------------------

class LambdaGNNDataset(Dataset):
    """
    SMILES + λmax の DataFrame から
    グラフデータセットを作るクラス。
    """

    def __init__(self, df: pd.DataFrame, smiles_col: str, target_col: str):
        super().__init__()
        self.smiles = df[smiles_col].astype(str).tolist()
        self.targets = df[target_col].astype(float).to_numpy()

        self.mols: List[Chem.Mol] = []
        self.valid_indices: List[int] = []

        for i, smi in enumerate(self.smiles):
            mol = smiles_to_mol(smi)
            if mol is not None:
                self.mols.append(mol)
                self.valid_indices.append(i)

        # ターゲットも valid のみに揃える
        self.targets = self.targets[self.valid_indices]

        print(f"[Dataset] original: {len(self.smiles)}  valid: {len(self.mols)}")

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx: int):
        mol = self.mols[idx]
        x, edge_index, edge_attr = mol_to_graph(mol)  # (N,F), (2,E), (E,Fe)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y,
        }


def collate_graphs(batch: List[dict]):
    """
    batch: list of
        {
            "x":         (Ni,F),
            "edge_index":(2,Ei),
            "edge_attr": (Ei,Fe),
            "y":         ()
        }

    -> x_batch:      (B, N_max, F)
       adj_batch:    (B, N_max, N_max)
       edge_feat_bt: (B, N_max, N_max, Fe)
       mask:         (B, N_max)
       y_batch:      (B,)
    """
    batch_size = len(batch)
    num_nodes_list = [b["x"].shape[0] for b in batch]
    N_max = max(num_nodes_list)
    feat_dim = batch[0]["x"].shape[1]
    edge_feat_dim = batch[0]["edge_attr"].shape[1] if batch[0]["edge_attr"].numel() > 0 else EDGE_FEAT_DIM

    x_batch = torch.zeros(batch_size, N_max, feat_dim, dtype=torch.float32)
    adj_batch = torch.zeros(batch_size, N_max, N_max, dtype=torch.float32)
    edge_feat_batch = torch.zeros(
        batch_size, N_max, N_max, edge_feat_dim, dtype=torch.float32
    )
    mask = torch.zeros(batch_size, N_max, dtype=torch.float32)
    y_batch = torch.zeros(batch_size, dtype=torch.float32)

    for i, item in enumerate(batch):
        x = item["x"]
        edge_index = item["edge_index"]
        edge_attr = item["edge_attr"]
        y = item["y"]

        n = x.shape[0]
        x_batch[i, :n, :] = x
        mask[i, :n] = 1.0
        y_batch[i] = y

        if edge_index.numel() > 0:
            src, dst = edge_index
            adj_batch[i, src, dst] = 1.0
            # エッジ特徴を対応する (src,dst) に配置
            for k in range(edge_attr.shape[0]):
                s = src[k].item()
                d = dst[k].item()
                edge_feat_batch[i, s, d, :] = edge_attr[k]

    return x_batch, adj_batch, edge_feat_batch, mask, y_batch