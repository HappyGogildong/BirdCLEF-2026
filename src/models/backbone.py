"""
src/models/backbone.py
───────────────────────
timm 기반 backbone 래퍼.

선택 전략
──────────
ㆍ tf_efficientnet_b2_ns (Noisy-Student)
    - ImageNet 1M 사전학습 → Mel-Spectrogram 이미지로 도메인 전이
    - BirdCLEF 과거 우승 솔루션에서 검증된 baseline
    - 파라미터 ~9.1M, 추론 속도 빠름 (Kaggle T4 기준 배치당 ~0.3s)

ㆍ convnext_small (대안)
    - 더 강한 표현력이 필요할 때 swap-in
    - 파라미터 ~50M

in_channels=1 처리:
    첫 conv 레이어의 weight 를 평균 내어 1채널로 변환 (pretrained 유지).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


SUPPORTED_BACKBONES = {
    "tf_efficientnet_b2_ns",
    "tf_efficientnet_b4_ns",
    "tf_efficientnetv2_s",
    "convnext_small",
    "convnext_tiny",
    "efficientformerv2_s2",
}


def _convert_to_single_channel(model: nn.Module, backbone_name: str) -> nn.Module:
    """첫 번째 Conv 레이어를 in_channels=1 로 변환.

    기존 RGB(3ch) weight를 채널 차원에서 평균 → 사전학습 가중치 유지.
    """
    if "efficientnet" in backbone_name:
        first_conv = model.conv_stem
    elif "convnext" in backbone_name:
        first_conv = model.stem[0]
    elif "efficientformer" in backbone_name:
        first_conv = model.patch_embed.proj[0]
    else:
        raise ValueError(f"Unknown backbone structure: {backbone_name}")

    orig_weight = first_conv.weight.data  # (out, 3, kH, kW)
    new_weight = orig_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(new_weight)
    if first_conv.bias is not None:
        new_conv.bias = nn.Parameter(first_conv.bias.data.clone())

    if "efficientnet" in backbone_name:
        model.conv_stem = new_conv
    elif "convnext" in backbone_name:
        model.stem[0] = new_conv
    elif "efficientformer" in backbone_name:
        model.patch_embed.proj[0] = new_conv

    return model


def build_backbone(name: str, pretrained: bool = True, drop_rate: float = 0.3) -> tuple[nn.Module, int]:
    """backbone 모델 + feature 차원을 반환.

    Returns
    -------
    (backbone, feature_dim)
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,          # head 제거, feature extractor로만 사용
        global_pool="avg",
        drop_rate=drop_rate,
    )
    model = _convert_to_single_channel(model, name)
    feature_dim = model.num_features

    return model, feature_dim
