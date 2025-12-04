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

# bond-type one-hot + conjugated + ring + stereo(3)
EDGE_FEAT_DIM = len(BOND_TYPE_LIST) + 2 + 3


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

    # 基本スカラー特徴
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

    # 価数 & 原子番号
    valence = float(atom.GetTotalValence())
    atomic_num = float(atom.GetAtomicNum())

    # H 原子数（置換基・局所環境）
    num_h = float(atom.GetTotalNumHs(includeNeighbors=True))

    # ring size 情報（5員環 / 6員環）
    in_5ring = float(atom.IsInRingSize(5))
    in_6ring = float(atom.IsInRingSize(6))

    # 立体化学（CIP R/S/その他）
    cip_R = 0.0
    cip_S = 0.0
    cip_other = 0.0
    if atom.HasProp("_CIPCode"):
        cip = atom.GetProp("_CIPCode")
        if cip == "R":
            cip_R = 1.0
        elif cip == "S":
            cip_S = 1.0
        else:
            cip_other = 1.0
    cip_one_hot = torch.tensor([cip_R, cip_S, cip_other], dtype=torch.float32)

    # 追加スカラー
    extra_scalars = torch.tensor(
        [valence, atomic_num, num_h, in_5ring, in_6ring],
        dtype=torch.float32,
    )

    feats = torch.cat(
        [
            one_hot,       # |ATOM_LIST|
            base_scalars,  # 4
            hyb_one_hot,   # |HYB_LIST|
            extra_scalars, # 5
            cip_one_hot,   # 3
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

    # 立体化学（cis/trans/その他）
    stereo = bond.GetStereo()
    is_cis = float(stereo == Chem.rdchem.BondStereo.STEREOCIS)
    is_trans = float(stereo == Chem.rdchem.BondStereo.STEREOTRANS)
    has_stereo = float(stereo != Chem.rdchem.BondStereo.STEREONONE)
    stereo_feats = torch.tensor([is_cis, is_trans, has_stereo], dtype=torch.float32)

    feats = torch.cat(
        [
            one_hot,                                 # bond type
            torch.tensor([conj, ring], dtype=torch.float32),
            stereo_feats,
        ],
        dim=0,
    )
    return feats  # (EDGE_FEAT_DIM,)


def mol_to_graph(
    mol: Chem.Mol,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RDKit Mol -> (node_feats, edge_index, edge_attr)

    node_feats: (N, F)
    edge_index: (2, E)  無向グラフなので双方向を格納
    edge_attr:  (E, EDGE_FEAT_DIM)
    """
    # ノード特徴
    node_feats_list: List[torch.Tensor] = []
    for atom in mol.GetAtoms():
        node_feats_list.append(atom_to_feat(atom))
    node_feats = torch.stack(node_feats_list, dim=0)  # (N, F)

    # エッジ
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


def collate_graphs(batch):
    """
    List[dict] -> batched グラフテンソル
    x_batch:        (B, N_max, F)
    adj_batch:      (B, N_max, N_max)
    edge_feat_batch:(B, N_max, N_max, Fe)
    mask:           (B, N_max)
    y_batch:        (B,)
    """
    xs = [b["x"] for b in batch]
    edge_indices = [b["edge_index"] for b in batch]
    edge_attrs = [b["edge_attr"] for b in batch]
    ys = [b["y"] for b in batch]

    B = len(xs)
    N_max = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]
    Fe = edge_attrs[0].shape[1] if edge_attrs[0].numel() > 0 else EDGE_FEAT_DIM

    x_batch = torch.zeros((B, N_max, F), dtype=torch.float32)
    adj_batch = torch.zeros((B, N_max, N_max), dtype=torch.float32)
    edge_feat_batch = torch.zeros((B, N_max, N_max, Fe), dtype=torch.float32)
    mask = torch.zeros((B, N_max), dtype=torch.float32)
    y_batch = torch.stack(ys, dim=0)

    for i, (x, edge_index, edge_attr) in enumerate(zip(xs, edge_indices, edge_attrs)):
        n = x.shape[0]
        x_batch[i, :n, :] = x
        mask[i, :n] = 1.0

        if edge_index.numel() > 0:
            src, dst = edge_index
            adj_batch[i, src, dst] = 1.0
            # エッジ特徴を対応する (src,dst) に配置
            for k in range(edge_attr.shape[0]):
                s = src[k].item()
                d = dst[k].item()
                edge_feat_batch[i, s, d, :] = edge_attr[k]

    return x_batch, adj_batch, edge_feat_batch, mask, y_batch