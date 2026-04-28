"""
src/training/loss.py
─────────────────────
BirdCLEF 2026 손실 함수 모듈.

선택 전략
──────────
다중 레이블 + 클래스 극심 불균형 → 단순 BCE 대신:

1. FocalBCELoss (기본)
   - Focal Loss 감마를 통해 쉬운 샘플(nocall) 기여도를 낮춤
   - pos_weight로 클래스 불균형 보정
   - label_smoothing으로 과신 방지

2. AsymmetricLoss (강화 옵션)
   - 양성/음성에 서로 다른 감마 적용
   - 음성 확률을 클리핑해 FP 패널티를 줄임
   - BirdCLEF 상위 솔루션에서 자주 사용

factory 함수 build_loss()를 통해 config 에서 선택.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Focal BCE Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """pos_weight + label smoothing + focal term 통합.

    Parameters
    ----------
    gamma         : focal 감마 (0=vanilla BCE, 2=강한 focal)
    label_smoothing: 양성 라벨을 1→(1-ε), 음성을 0→ε 으로 완화
    pos_weight    : 텐서 (n_classes,) — 클래스별 양성 가중치
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)
        targets : (B, C)  — float {0, 1} or soft
        """
        # Label smoothing
        smooth_targets = targets * (1 - self.ls) + (1 - targets) * self.ls

        # BCE (element-wise)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smooth_targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Focal weight
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)

        loss = (focal_weight * bce).mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# 2. Asymmetric Loss (ASL)
# ──────────────────────────────────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021 — Asymmetric Loss for Multi-Label Classification.

    양성 클래스: gamma_pos=0 (focal 없음, 쉬운 양성도 중요)
    음성 클래스: gamma_neg=4 (쉬운 음성 = 대부분 nocall → 패널티 크게 감소)
    clip        : 음성 확률을 clip 이상으로만 취급 (FP 억제)
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # 음성 확률 클리핑 (시프트 트릭)
        probs_neg = (probs + self.clip).clamp(max=1.0)

        # log 확률
        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1 - probs_neg).clamp(min=self.eps))

        # Focal 가중치
        pos_loss = -targets * (1 - probs).pow(self.gamma_pos) * log_pos
        neg_loss = -(1 - targets) * probs_neg.pow(self.gamma_neg) * log_neg

        loss = (pos_loss + neg_loss).mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_loss(loss_cfg, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """config.training.loss 설정에 따라 손실 함수를 반환.

    loss_cfg.name 선택지:
        'bce'          → 기본 BCEWithLogitsLoss + pos_weight
        'bce_focal'    → FocalBCELoss (기본 추천)
        'asymmetric'   → AsymmetricLoss
    """
    name = loss_cfg.name.lower()

    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif name == "bce_focal":
        return FocalBCELoss(
            gamma=loss_cfg.focal_gamma,
            label_smoothing=loss_cfg.label_smoothing,
            pos_weight=pos_weight,
        )

    elif name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=0.0,
            gamma_neg=loss_cfg.get("gamma_neg", 4.0),
            clip=loss_cfg.get("clip", 0.05),
        )

    else:
        raise ValueError(f"Unknown loss: {name}. Choose from [bce, bce_focal, asymmetric]")
