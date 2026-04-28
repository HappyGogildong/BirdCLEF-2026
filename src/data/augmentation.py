"""
src/data/augmentation.py
─────────────────────────
BirdCLEF 2026용 오디오 증강 모듈.

단계별 증강 전략
────────────────
1. 파형(Waveform) 단계
   - 가우시안 잡음 주입
   - 배경 환경음 믹싱 (SNR 기반)
   - Mixup: 최대 3개 샘플 가중 합산 (라벨도 동시 처리)

2. 스펙트로그램(Spectrogram) 단계
   - SpecAugment: 시간·주파수 마스킹
   - 랜덤 롤(시간축 이동)
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 파형 단계 증강
# ──────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(
    waveform: np.ndarray,
    min_amplitude: float = 0.001,
    max_amplitude: float = 0.015,
    p: float = 0.3,
) -> np.ndarray:
    if random.random() < p:
        amp = random.uniform(min_amplitude, max_amplitude)
        noise = np.random.randn(*waveform.shape).astype(np.float32) * amp
        waveform = waveform + noise
    return waveform


def mix_background(
    waveform: np.ndarray,
    background_pool: List[np.ndarray],
    min_snr_db: float = 3.0,
    max_snr_db: float = 30.0,
    p: float = 0.3,
) -> np.ndarray:
    """background_pool 에서 무작위로 배경음 1개를 선택해 SNR 기준으로 믹스."""
    if not background_pool or random.random() >= p:
        return waveform

    bg = random.choice(background_pool)
    # 길이 맞추기
    if len(bg) < len(waveform):
        repeats = int(np.ceil(len(waveform) / len(bg)))
        bg = np.tile(bg, repeats)
    start = random.randint(0, len(bg) - len(waveform))
    bg = bg[start: start + len(waveform)]

    signal_rms = np.sqrt(np.mean(waveform ** 2) + 1e-9)
    bg_rms = np.sqrt(np.mean(bg ** 2) + 1e-9)
    snr_db = random.uniform(min_snr_db, max_snr_db)
    target_bg_rms = signal_rms / (10 ** (snr_db / 20.0))
    bg = bg * (target_bg_rms / bg_rms)
    return waveform + bg


class WaveformMixup:
    """멀티-소스 Mixup (최대 max_n_mix 개 샘플 혼합, 라벨 동시 처리).

    사용법:
        mixer = WaveformMixup(alpha=0.4, max_n_mix=3, p=0.4)
        new_wave, new_label = mixer(wave, label, other_waves, other_labels)
    """

    def __init__(self, alpha: float = 0.4, max_n_mix: int = 3, p: float = 0.4):
        self.alpha = alpha
        self.max_n_mix = max_n_mix
        self.p = p

    def __call__(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        pool_waves: List[np.ndarray],
        pool_labels: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p or not pool_waves:
            return waveform, label

        n_mix = min(random.randint(1, self.max_n_mix), len(pool_waves))
        indices = random.sample(range(len(pool_waves)), n_mix)

        lambdas = np.random.dirichlet([self.alpha] * (n_mix + 1))
        mixed_wave = lambdas[0] * waveform
        mixed_label = lambdas[0] * label.astype(np.float32)

        for k, idx in enumerate(indices):
            src = pool_waves[idx]
            lbl = pool_labels[idx].astype(np.float32)
            # 길이 정렬
            if len(src) < len(waveform):
                src = np.pad(src, (0, len(waveform) - len(src)), mode="wrap")
            else:
                src = src[: len(waveform)]
            mixed_wave = mixed_wave + lambdas[k + 1] * src
            mixed_label = mixed_label + lambdas[k + 1] * lbl

        mixed_label = np.clip(mixed_label, 0.0, 1.0)
        return mixed_wave.astype(np.float32), mixed_label


# ──────────────────────────────────────────────────────────────────────────────
# 스펙트로그램 단계 증강
# ──────────────────────────────────────────────────────────────────────────────

def time_masking(
    mel: np.ndarray,
    max_mask: int = 20,
    n_masks: int = 2,
    p: float = 0.5,
) -> np.ndarray:
    """시간 축(W) 마스킹 (0으로 채움)."""
    if random.random() >= p:
        return mel
    mel = mel.copy()
    _, t = mel.shape
    for _ in range(n_masks):
        mask_len = random.randint(0, min(max_mask, t))
        start = random.randint(0, t - mask_len)
        mel[:, start: start + mask_len] = 0.0
    return mel


def freq_masking(
    mel: np.ndarray,
    max_mask: int = 20,
    n_masks: int = 2,
    p: float = 0.5,
) -> np.ndarray:
    """주파수 축(H) 마스킹."""
    if random.random() >= p:
        return mel
    mel = mel.copy()
    h, _ = mel.shape
    for _ in range(n_masks):
        mask_len = random.randint(0, min(max_mask, h))
        start = random.randint(0, h - mask_len)
        mel[start: start + mask_len, :] = 0.0
    return mel


def random_time_roll(mel: np.ndarray, p: float = 0.3) -> np.ndarray:
    """시간 축 랜덤 순환 이동 (경계 아티팩트를 피하기 위해 roll 사용)."""
    if random.random() >= p:
        return mel
    shift = random.randint(0, mel.shape[1] - 1)
    return np.roll(mel, shift, axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 통합 증강 파이프라인
# ──────────────────────────────────────────────────────────────────────────────

class TrainAugmentation:
    """학습 시 파형 + 스펙트로그램 증강을 순서대로 적용하는 통합 객체."""

    def __init__(
        self,
        cfg,  # OmegaConf DictConfig.augmentation.train
        background_pool: Optional[List[np.ndarray]] = None,
        mixup_helper: Optional[WaveformMixup] = None,
    ):
        self.cfg = cfg
        self.background_pool = background_pool or []
        self.mixup_helper = mixup_helper

    def apply_waveform(
        self,
        waveform: np.ndarray,
        label: np.ndarray,
        pool_waves: Optional[List[np.ndarray]] = None,
        pool_labels: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. 가우시안 잡음
        c = self.cfg.add_gaussian_noise
        waveform = add_gaussian_noise(
            waveform,
            min_amplitude=c.min_amplitude,
            max_amplitude=c.max_amplitude,
            p=c.p,
        )
        # 2. 배경 잡음
        c = self.cfg.background_noise
        waveform = mix_background(
            waveform,
            self.background_pool,
            min_snr_db=c.min_snr_db,
            max_snr_db=c.max_snr_db,
            p=c.p,
        )
        # 3. Mixup
        if self.mixup_helper and pool_waves and pool_labels:
            waveform, label = self.mixup_helper(
                waveform, label, pool_waves, pool_labels
            )
        return waveform, label

    def apply_spectrogram(self, mel: np.ndarray) -> np.ndarray:
        c = self.cfg
        mel = time_masking(mel, max_mask=c.time_masking.max_time_mask, p=c.time_masking.p)
        mel = freq_masking(mel, max_mask=c.freq_masking.max_freq_mask, p=c.freq_masking.p)
        mel = random_time_roll(mel, p=0.3)
        return mel
