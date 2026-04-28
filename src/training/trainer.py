"""
src/training/trainer.py
────────────────────────
학습(train) / 검증(validate) 루프.

주요 기능
─────────
- Gradient accumulation (effective batch size 제어)
- Mixed Precision (torch.cuda.amp)
- Gradient clipping
- 에폭별 macro-ROC-AUC 계산 및 출력
- Best checkpoint 저장
- WandB 로깅 (선택)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import macro_roc_auc_score


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        cfg,
        device: torch.device,
        output_dir: str | Path,
        fold: int = 0,
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fold = fold
        self.wandb_run = wandb_run

        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.best_auc = 0.0
        self.best_epoch = 0

    # ──────────────────────────────────────────────────────────────────────────
    # 학습 한 에폭
    # ──────────────────────────────────────────────────────────────────────────

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        t_cfg = self.cfg.training
        accum = t_cfg.accumulation_steps
        total_loss = 0.0
        self.optimizer.zero_grad()
        bar = tqdm(loader, desc=f"[Fold {self.fold}] Epoch {epoch} Train", leave=False)

        for step, batch in enumerate(bar):
            specs = batch["spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(specs)
                loss = self.criterion(logits, labels) / accum

            self.scaler.scale(loss).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=t_cfg.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum
            bar.set_postfix(loss=f"{loss.item() * accum:.4f}")

        avg_loss = total_loss / len(loader)
        return avg_loss

    # ──────────────────────────────────────────────────────────────────────────
    # 검증 한 에폭
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        bar = tqdm(loader, desc=f"[Fold {self.fold}] Epoch {epoch} Valid", leave=False)

        for batch in bar:
            specs = batch["spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(specs)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        auc = macro_roc_auc_score(labels, preds)
        avg_loss = total_loss / len(loader)
        return avg_loss, auc

    # ──────────────────────────────────────────────────────────────────────────
    # 전체 학습 루프
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> float:
        n_epochs = self.cfg.training.epochs
        print(f"\n{'='*60}")
        print(f"  Fold {self.fold} 학습 시작 | 총 {n_epochs} 에폭")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_loss = self.train_one_epoch(train_loader, epoch)
            valid_loss, valid_auc = self.validate(valid_loader, epoch)

            # Scheduler step
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Loss(T/V): {train_loss:.4f}/{valid_loss:.4f} | "
                f"AUC: {valid_auc:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # WandB 로깅
            if self.wandb_run:
                self.wandb_run.log({
                    f"fold{self.fold}/train_loss": train_loss,
                    f"fold{self.fold}/valid_loss": valid_loss,
                    f"fold{self.fold}/valid_auc": valid_auc,
                    f"fold{self.fold}/lr": lr,
                    "epoch": epoch,
                })

            # Best model 저장
            if valid_auc > self.best_auc:
                self.best_auc = valid_auc
                self.best_epoch = epoch
                ckpt_path = self.output_dir / f"fold{self.fold}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "auc": valid_auc,
                    },
                    ckpt_path,
                )
                print(f"  ✓ 체크포인트 저장 → {ckpt_path}  (AUC={valid_auc:.4f})")

        print(f"\n[Fold {self.fold}] 최고 AUC: {self.best_auc:.4f}  (epoch {self.best_epoch})")
        return self.best_auc
