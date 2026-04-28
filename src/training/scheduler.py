"""
src/training/scheduler.py
──────────────────────────
Optimizer / Scheduler 빌더.

전략
────
Optimizer : AdamW
  - weight_decay=0.01 (bias/BN 파라미터는 decay 제외)
  - backbone lr를 head lr의 0.1× (layer-wise lr decay)

Scheduler : Cosine Annealing with Warm-up
  - 초기 warmup_epochs 동안 lr를 선형으로 0 → base_lr
  - 이후 cosine curve로 min_lr까지 감소
  - T_max=epochs, eta_min=min_lr
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def _get_param_groups(model: nn.Module, base_lr: float, backbone_lr_scale: float = 0.1):
    """backbone 파라미터와 head 파라미터에 서로 다른 lr 적용."""
    backbone_params = []
    head_params = []
    no_decay = {"bias", "LayerNorm.weight", "BatchNorm", "bn"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("backbone") or name.startswith("gem")
        is_no_decay = any(nd in name for nd in no_decay)

        if is_backbone:
            group = {"params": param, "lr": base_lr * backbone_lr_scale}
        else:
            group = {"params": param, "lr": base_lr}

        if is_no_decay:
            group["weight_decay"] = 0.0
        backbone_params.append(group) if is_backbone else head_params.append(group)

    return backbone_params + head_params


def build_optimizer(model: nn.Module, opt_cfg, backbone_lr_scale: float = 0.1) -> AdamW:
    param_groups = _get_param_groups(model, opt_cfg.lr, backbone_lr_scale)
    optimizer = AdamW(
        param_groups,
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
    )
    return optimizer


def build_scheduler(optimizer, sch_cfg, epochs: int):
    """Warmup + CosineAnnealing 조합 스케줄러."""
    warmup_epochs = sch_cfg.warmup_epochs
    min_lr = sch_cfg.min_lr
    base_lr = optimizer.param_groups[-1]["lr"]  # head lr 기준

    # Warmup: 선형 증가 (0 → base_lr)
    def warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0  # CosineAnnealingLR 이 이후를 담당

    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler
