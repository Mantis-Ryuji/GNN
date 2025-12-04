from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    # 入力データ
    csv_path: str = "datasets/dataset.csv"  # smiles, lambda_max を含む CSV
    smiles_col: str = "SMILES"
    target_col: str = "λmax"

    # データ分割（train / val のみ）
    val_ratio: float = 0.2

    # モデル / 学習ハイパラ
    batch_size: int = 1024
    num_epochs: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 512
    num_layers: int = 5
    dropout: float = 0.1

    # scheduler (ReduceLROnPlateau)
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6

    # early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # その他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"