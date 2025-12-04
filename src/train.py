from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import TrainConfig
from data import LambdaGNNDataset, collate_graphs
from model import GINERegressor


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(pred, target))


class ModelEMA:
    """
    簡易 EMA 実装:
      - train では生 model を更新
      - 各ステップ後に ema_state = decay * ema_state + (1-decay) * model_param
      - eval / best 保存は EMA の重みで行う
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_state: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        d = self.decay
        for k, v in msd.items():
            if k in self.ema_state:
                self.ema_state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.ema_state[k] = v.detach().clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.ema_state

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        self.ema_state = {k: v.clone() for k, v in state.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: ModelEMA | None = None,
    ema_start_step: int = 0,
):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_samples = 0
    global_step = 0

    for x, adj, edge_attr, mask, y in loader:
        x = x.to(device)
        adj = adj.to(device)
        edge_attr = edge_attr.to(device)
        mask = mask.to(device)
        y = y.to(device)

        pred = model(x, adj, edge_attr, mask)
        loss = nn.functional.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA 更新
        if ema is not None and global_step >= ema_start_step:
            ema.update(model)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_rmse += rmse_loss(pred, y).item() * batch_size
        n_samples += batch_size

        global_step += 1

    return total_loss / n_samples, total_rmse / n_samples


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ema: ModelEMA | None = None,
):
    """
    評価は基本 EMA の重みで行う。
    ema=None のときは生 model をそのまま評価。
    """
    if ema is not None:
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())

    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_samples = 0

    for x, adj, edge_attr, mask, y in loader:
        x = x.to(device)
        adj = adj.to(device)
        edge_attr = edge_attr.to(device)
        mask = mask.to(device)
        y = y.to(device)

        pred = model(x, adj, edge_attr, mask)
        loss = nn.functional.mse_loss(pred, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_rmse += rmse_loss(pred, y).item() * batch_size
        n_samples += batch_size

    if ema is not None:
        model.load_state_dict(backup_state)

    return total_loss / n_samples, total_rmse / n_samples


class EarlyStopping:
    """
    val_metric (ここでは val_rmse) が改善しなくなったら学習を打ち切る。
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        metric: 小さいほど良い値 (例: rmse)
        戻り値: True のとき early stop すべき
        """
        if metric < self.best - self.min_delta:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint_last(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau,
    best_val_rmse: float,
    best_state: dict | None,
    ema_state: dict | None,
):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_rmse": best_val_rmse,
        "best_state": best_state,
        "ema_state": ema_state,
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)


def save_checkpoint_best(
    path: str,
    best_epoch: int,
    best_state: dict,
    best_val_rmse: float,
):
    ckpt = {
        "epoch": best_epoch,
        "model": best_state,
        "best_val_rmse": best_val_rmse,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau,
):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    rng_state = ckpt.get("rng_state", None)
    if rng_state is not None:
        torch.set_rng_state(rng_state["cpu"])
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state(rng_state["cuda"])

    start_epoch = ckpt["epoch"] + 1
    best_val_rmse = ckpt.get("best_val_rmse", math.inf)
    best_state = ckpt.get("best_state", None)
    best_epoch = ckpt.get("best_epoch", start_epoch - 1)
    ema_state = ckpt.get("ema_state", None)

    return start_epoch, best_val_rmse, best_state, best_epoch, ema_state


def main(resume: bool = True, resume_path: str = "src/runs/last.pt"):
    cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs("src/runs", exist_ok=True)

    # ---- データ読み込み ----
    df = pd.read_csv(cfg.csv_path)

    # ===== y(λmax) の標準化 =====
    y_raw = df[cfg.target_col].astype(float)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std())

    if y_std <= 0.0:
        raise ValueError(
            f"Target std is non-positive (std={y_std}). "
            "標準化できません。データを確認してください。"
        )

    df[cfg.target_col] = (y_raw - y_mean) / y_std

    # 推論用に mean / std を保存
    torch.save({"mean": y_mean, "std": y_std}, "src/runs/target_scaler.pt")
    print(
        f"[TargetScaler] mean={y_mean:.4f}, std={y_std:.4f} "
        "-> saved to src/runs/target_scaler.pt"
    )

    dataset = LambdaGNNDataset(df, cfg.smiles_col, cfg.target_col)

    # ---- train / val split ----
    N = len(dataset)
    n_val = int(N * cfg.val_ratio)
    n_train = N - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print(f"[Split] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # ---- モデル構築 ----
    sample = dataset[0]
    in_dim = sample["x"].shape[1]
    edge_dim = sample["edge_attr"].shape[1]
    print(f"[Model] in_dim={in_dim}  edge_dim={edge_dim}")

    device = torch.device(cfg.device)
    model = GINERegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
        verbose=True,
    )

    # ---- EMA ----
    ema_decay = 0.999
    ema = ModelEMA(model, decay=ema_decay)
    ema_start_step = 0  # 必要ならここで遅らせる

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )

    # ---- resume ----
    start_epoch = 1
    best_val_rmse = math.inf
    best_state: dict | None = None
    best_epoch = 0

    if resume and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        (
            start_epoch,
            best_val_rmse,
            best_state,
            best_epoch,
            ema_state,
        ) = load_checkpoint(resume_path, model, optimizer, scheduler)
        early_stopper.best = best_val_rmse
        if ema_state is not None:
            ema.load_state_dict(ema_state)
        print(
            f"[Resume] start_epoch={start_epoch}  "
            f"best_val_rmse={best_val_rmse:.3f}  best_epoch={best_epoch}"
        )
    elif resume:
        print(f"[Resume] Checkpoint not found at {resume_path}, start from scratch.")

    # ---- 学習ループ ----
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_loss, train_rmse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ema=ema,
            ema_start_step=ema_start_step,
        )
        # val は EMA 重みで評価
        val_loss, val_rmse = eval_epoch(model, val_loader, device, ema=ema)

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        # 元スケール RMSE を見たい場合は val_rmse * y_std で変換
        val_rmse_orig = val_rmse * y_std
        train_rmse_orig = train_rmse * y_std

        print(
            f"[Epoch {epoch:03d}] "
            f"lr={current_lr:.3e}  "
            f"train_loss={train_loss:.4f}  train_rmse(z)={train_rmse:.3f}  "
            f"val_loss={val_loss:.4f}  val_rmse(z)={val_rmse:.3f}  "
            f"train_rmse(orig)={train_rmse_orig:.3f}  val_rmse(orig)={val_rmse_orig:.3f}"
        )

        # ベストモデル更新（EMA 重みを保存）
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu() for k, v in ema.state_dict().items()}
            best_epoch = epoch
            save_checkpoint_best(
                path="src/runs/best.pt",
                best_epoch=best_epoch,
                best_state=best_state,
                best_val_rmse=best_val_rmse,
            )
            print(
                f"[BestUpdate] epoch={epoch}  "
                f"val_rmse(z)={best_val_rmse:.3f}  "
                f"val_rmse(orig)={best_val_rmse * y_std:.3f}  -> saved src/runs/best.pt"
            )

        # last checkpoint 保存（生 model + EMA）
        save_checkpoint_last(
            path="src/runs/last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_rmse=best_val_rmse,
            best_state=best_state,
            ema_state=ema.state_dict(),
        )
        print(f"[Checkpoint] Saved src/runs/last.pt (epoch={epoch})")

        if early_stopper.step(val_rmse):
            print(
                f"[EarlyStop] epoch={epoch}  "
                f"best_val_rmse(z)={best_val_rmse:.3f}  "
                f"best_val_rmse(orig)={best_val_rmse * y_std:.3f} "
                f"(best_epoch={best_epoch})"
            )
            break

    # ---- best(EMA) を model に反映 ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[Best] epoch={best_epoch}  "
            f"val_rmse(z)={best_val_rmse:.3f}  "
            f"val_rmse(orig)={best_val_rmse * y_std:.3f}"
        )
    else:
        print("[Best] best_state is None (no improvement over inf?)")


if __name__ == "__main__":
    main(resume=True, resume_path="src/runs/last.pt")