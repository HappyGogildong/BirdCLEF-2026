"""
src/models/classifier.py
─────────────────────────
BirdCLEF 2026 최종 분류 모델.

아키텍처
─────────
Input: Mel-Spectrogram (B, 1, 128, T)
  ↓
Backbone (EfficientNet-B2-NS) → feature (B, 1408)
  ↓
Head: BN → Dropout → Linear(1408→512) → GELU → Dropout → Linear(512→234)
  ↓
Output: logits (B, 234)  — 학습 시 BCEWithLogitsLoss 적용

GeM Pooling (Power Mean Pooling) を使用することで
SpecAugment으로 마스킹된 영역에 덜 민감하게 동작.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import build_backbone


class GeM(nn.Module):
    """Generalized Mean Pooling.

    p=1  → 평균 풀링
    p=∞  → 최대 풀링
    p=3  ← 기본값 (시각·음향 태스크 경험적 최적)
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        p = self.p.clamp(min=1.0)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1).pow(1.0 / p)


class AttentionPool(nn.Module):
    """Channel-wise attention pooling (가벼운 SE-like head).

    frequency-axis 에서 중요한 band를 동적으로 선택.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  — frame 단위 feature
        w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * w).sum(dim=1)               # (B, C)


class BirdClefClassifier(nn.Module):
    """전체 BirdCLEF 분류 모델.

    Parameters
    ----------
    model_cfg  : config.model (OmegaConf)
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.backbone, feat_dim = build_backbone(
            name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            drop_rate=model_cfg.drop_rate,
        )
        # timm의 기본 global_pool을 비활성화하고 커스텀 풀링 사용
        self.gem = GeM(p=3.0, trainable=True)

        hidden = model_cfg.head_hidden_dim
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(model_cfg.drop_rate),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(model_cfg.drop_rate * 0.5),
            nn.Linear(hidden, model_cfg.num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """backbone feature map 반환 (B, C, H, W)."""
        # timm 모델의 forward_features는 global_pool 전까지 실행
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, n_mels, time_frames)

        Returns
        -------
        logits : (B, n_classes)
        """
        feat_map = self.forward_features(x)      # (B, C, H', W')
        pooled = self.gem(feat_map).flatten(1)   # (B, C)
        logits = self.head(pooled)               # (B, n_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """추론용: sigmoid 적용한 확률 반환."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))
