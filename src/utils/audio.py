"""
src/utils/audio.py
──────────────────
오디오 로딩·변환·멜-스펙트로그램 생성 공통 유틸리티.
모든 파이프라인에서 공유하는 단일 진입점으로 유지한다.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# 파형 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    sr: int = 32_000,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """OGG / WAV / FLAC 파일을 로드하여 numpy 배열로 반환.

    Returns
    -------
    waveform : np.ndarray, shape (n_samples,)  — mono 기준
    sr       : int
    """
    try:
        waveform, orig_sr = sf.read(str(path), always_2d=True)
        # soundfile은 (samples, channels) 로 반환
        waveform = waveform.T  # → (channels, samples)
        if mono:
            waveform = waveform.mean(axis=0)

        if orig_sr != sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sr)

        # offset / duration 적용
        start = int(offset * sr)
        if duration is not None:
            end = start + int(duration * sr)
            waveform = waveform[start:end]
        else:
            waveform = waveform[start:]

    except Exception:
        # 에러 시 묵음 반환 (파이프라인 중단 방지)
        samples = int((duration or 5.0) * sr)
        waveform = np.zeros(samples, dtype=np.float32)

    return waveform.astype(np.float32), sr


def pad_or_trim(waveform: np.ndarray, target_len: int, mode: str = "wrap") -> np.ndarray:
    """파형 길이를 target_len 샘플로 맞춘다.

    Parameters
    ----------
    mode : 'wrap' | 'constant'
        짧을 때 반복(wrap) 또는 0-패딩(constant)
    """
    n = len(waveform)
    if n == target_len:
        return waveform
    if n > target_len:
        return waveform[:target_len]
    # 짧은 경우
    if mode == "wrap":
        repeats = int(np.ceil(target_len / n))
        return np.tile(waveform, repeats)[:target_len]
    return np.pad(waveform, (0, target_len - n), mode="constant")


def normalize_waveform(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS 기준으로 파형의 dBFS를 target_db 로 정규화."""
    rms = np.sqrt(np.mean(waveform ** 2) + 1e-9)
    target_rms = 10 ** (target_db / 20.0)
    return waveform * (target_rms / rms)


# ──────────────────────────────────────────────────────────────────────────────
# 멜-스펙트로그램 변환
# ──────────────────────────────────────────────────────────────────────────────

def compute_melspectrogram(
    waveform: np.ndarray,
    sr: int = 32_000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 14_000.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """파형 → log-mel spectrogram.

    Returns
    -------
    mel : np.ndarray, shape (n_mels, time_frames), dtype float32
          값 범위 [0, 1]로 정규화
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    # 파워 → 데시벨, 상대 정규화
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    # [−top_db, 0] → [0, 1]
    log_mel = (log_mel + top_db) / top_db
    return log_mel.astype(np.float32)


def melspec_to_tensor(mel: np.ndarray) -> torch.Tensor:
    """(H, W) → (1, H, W) float32 tensor."""
    return torch.from_numpy(mel).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# 음향경관(soundscape) 5초 분할 유틸
# ──────────────────────────────────────────────────────────────────────────────

def split_soundscape(
    path: str | Path,
    sr: int = 32_000,
    window_sec: float = 5.0,
    step_sec: float = 5.0,
) -> list[Tuple[float, np.ndarray]]:
    """1분 녹음을 (offset, 파형) 리스트로 반환.

    Returns
    -------
    chunks : list of (start_sec, waveform_array)
    """
    waveform, _ = load_audio(path, sr=sr, mono=True)
    total_sec = len(waveform) / sr
    window_len = int(window_sec * sr)
    step_len = int(step_sec * sr)
    chunks = []

    start = 0
    t = 0.0
    while start + window_len <= len(waveform):
        chunk = waveform[start: start + window_len]
        chunks.append((t, chunk))
        start += step_len
        t += step_sec

    return chunks
