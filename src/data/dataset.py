"""
src/data/dataset.py
────────────────────
BirdCLEF 2026용 PyTorch Dataset 구현.

BirdClefTrainDataset  : train_audio 짧은 클립 기반 학습
SoundscapeDataset     : train_soundscapes 5초 단위 학습 (라벨 있음)
TestSoundscapeDataset : test_soundscapes 5초 단위 추론
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio import (
    compute_melspectrogram,
    load_audio,
    melspec_to_tensor,
    pad_or_trim,
    normalize_waveform,
    split_soundscape,
)
from src.data.augmentation import TrainAugmentation


# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _mel_cfg_kwargs(audio_cfg) -> dict:
    return dict(
        sr=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        n_mels=audio_cfg.n_mels,
        fmin=audio_cfg.fmin,
        fmax=audio_cfg.fmax,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. 짧은 클립 학습 데이터셋 (train_audio/)
# ──────────────────────────────────────────────────────────────────────────────

class BirdClefTrainDataset(Dataset):
    """train_audio 디렉터리에 있는 짧은 클립(.ogg)을 5초 단위로 잘라 학습.

    Parameters
    ----------
    df          : split_kfold() 적용된 train_metadata DataFrame
    audio_dir   : train_audio/ 절대 경로
    label2idx   : 종 코드 → 정수 인덱스
    audio_cfg   : config.audio (OmegaConf)
    is_train    : True=학습, False=검증
    augmentation: TrainAugmentation 인스턴스 (학습 시만 사용)
    mixup_pool  : Mixup용 (waveform, label) 튜플 리스트 (학습 시 제공)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str | Path,
        label2idx: Dict[str, int],
        audio_cfg,
        is_train: bool = True,
        augmentation: Optional[TrainAugmentation] = None,
        mixup_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label2idx = label2idx
        self.n_classes = len(label2idx)
        self.audio_cfg = audio_cfg
        self.is_train = is_train
        self.aug = augmentation
        self.mixup_pool = mixup_pool or []
        self.target_len = int(audio_cfg.sample_rate * audio_cfg.clip_duration)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_label(self, row: pd.Series) -> np.ndarray:
        vec = np.zeros(self.n_classes, dtype=np.float32)
        # primary label
        idx = self.label2idx.get(row["primary_label"])
        if idx is not None:
            vec[idx] = 1.0
        # secondary labels (문자열 파싱)
        sec = row.get("secondary_labels", "")
        if isinstance(sec, str) and sec not in ("", "[]"):
            sec = sec.strip("[]").replace("'", "").replace('"', "")
            for sp in sec.split(","):
                sp = sp.strip()
                sidx = self.label2idx.get(sp)
                if sidx is not None:
                    vec[sidx] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        # 파일 경로: train_audio/species_code/XC12345.ogg
        audio_path = self.audio_dir / row["filename"]
        sr = self.audio_cfg.sample_rate

        waveform, _ = load_audio(str(audio_path), sr=sr, mono=True)
        # 길이 정규화
        waveform = pad_or_trim(waveform, self.target_len, mode="wrap")
        waveform = normalize_waveform(waveform, target_db=self.audio_cfg.target_db)

        label = self._encode_label(row)

        # ── 파형 증강 (학습 시) ──
        if self.is_train and self.aug is not None:
            pool_w = [p[0] for p in self.mixup_pool]
            pool_l = [p[1] for p in self.mixup_pool]
            waveform, label = self.aug.apply_waveform(waveform, label, pool_w, pool_l)

        # ── 멜-스펙트로그램 ──
        mel = compute_melspectrogram(waveform, **_mel_cfg_kwargs(self.audio_cfg))

        # ── 스펙트로그램 증강 (학습 시) ──
        if self.is_train and self.aug is not None:
            mel = self.aug.apply_spectrogram(mel)

        tensor = melspec_to_tensor(mel)  # (1, H, W)
        return {"spectrogram": tensor, "label": torch.tensor(label, dtype=torch.float32)}


# ──────────────────────────────────────────────────────────────────────────────
# 2. 음향경관 학습 데이터셋 (train_soundscapes/ + labels)
# ──────────────────────────────────────────────────────────────────────────────

class SoundscapeDataset(Dataset):
    """train_soundscape_labels.csv 기반 5초 단위 다중 라벨 데이터셋.

    각 행 = (audio_id, seconds) 조합으로 정확히 하나의 5초 구간.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,       # train_soundscape_labels.csv
        soundscape_dir: str | Path,
        label2idx: Dict[str, int],
        audio_cfg,
        is_train: bool = False,
        augmentation: Optional[TrainAugmentation] = None,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.soundscape_dir = Path(soundscape_dir)
        self.label2idx = label2idx
        self.n_classes = len(label2idx)
        self.audio_cfg = audio_cfg
        self.is_train = is_train
        self.aug = augmentation
        self.target_len = int(audio_cfg.sample_rate * audio_cfg.clip_duration)

        # 각 파일의 파형을 캐싱해 반복 로딩을 방지
        self._cache: Dict[str, np.ndarray] = {}

    def _load_cached(self, audio_id: str) -> np.ndarray:
        if audio_id not in self._cache:
            path = self.soundscape_dir / f"{audio_id}.ogg"
            wav, _ = load_audio(str(path), sr=self.audio_cfg.sample_rate, mono=True)
            self._cache[audio_id] = wav
        return self._cache[audio_id]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_id = str(row["audio_id"])
        end_sec = float(row["seconds"])
        start_sec = end_sec - self.audio_cfg.clip_duration
        sr = self.audio_cfg.sample_rate

        full_wav = self._load_cached(audio_id)
        start_sample = max(0, int(start_sec * sr))
        end_sample = start_sample + self.target_len
        chunk = full_wav[start_sample:end_sample]
        chunk = pad_or_trim(chunk, self.target_len, mode="constant")
        chunk = normalize_waveform(chunk, target_db=self.audio_cfg.target_db)

        # 라벨 인코딩
        species_str = row.get("species", "nocall")
        vec = np.zeros(self.n_classes, dtype=np.float32)
        if isinstance(species_str, str) and species_str.lower() != "nocall":
            for sp in species_str.strip().split():
                sidx = self.label2idx.get(sp)
                if sidx is not None:
                    vec[sidx] = 1.0

        mel = compute_melspectrogram(chunk, **_mel_cfg_kwargs(self.audio_cfg))
        if self.is_train and self.aug is not None:
            mel = self.aug.apply_spectrogram(mel)

        tensor = melspec_to_tensor(mel)
        return {
            "spectrogram": tensor,
            "label": torch.tensor(vec, dtype=torch.float32),
            "row_id": str(row["row_id"]),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. 테스트 추론 데이터셋 (test_soundscapes/)
# ──────────────────────────────────────────────────────────────────────────────

class TestSoundscapeDataset(Dataset):
    """테스트용 1분 녹음을 5초 단위로 분할해 추론.

    row_id 컨벤션: '{audio_id}_{end_seconds}'  (sample_submission.csv 준수)
    """

    def __init__(
        self,
        submission_df: pd.DataFrame,   # sample_submission.csv
        soundscape_dir: str | Path,
        audio_cfg,
    ):
        # row_id 목록 추출
        self.row_ids: List[str] = submission_df["row_id"].tolist()
        self.soundscape_dir = Path(soundscape_dir)
        self.audio_cfg = audio_cfg
        self.target_len = int(audio_cfg.sample_rate * audio_cfg.clip_duration)
        self._cache: Dict[str, np.ndarray] = {}

    def _parse_row_id(self, row_id: str) -> Tuple[str, float]:
        """'XC123456_10' → ('XC123456', 10.0)"""
        parts = row_id.rsplit("_", 1)
        return parts[0], float(parts[1])

    def _load_cached(self, audio_id: str) -> np.ndarray:
        if audio_id not in self._cache:
            path = self.soundscape_dir / f"{audio_id}.ogg"
            wav, _ = load_audio(str(path), sr=self.audio_cfg.sample_rate, mono=True)
            self._cache[audio_id] = wav
        return self._cache[audio_id]

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row_id = self.row_ids[idx]
        audio_id, end_sec = self._parse_row_id(row_id)
        start_sec = end_sec - self.audio_cfg.clip_duration
        sr = self.audio_cfg.sample_rate

        full_wav = self._load_cached(audio_id)
        start_sample = max(0, int(start_sec * sr))
        chunk = full_wav[start_sample: start_sample + self.target_len]
        chunk = pad_or_trim(chunk, self.target_len, mode="constant")
        chunk = normalize_waveform(chunk, target_db=self.audio_cfg.target_db)

        mel = compute_melspectrogram(chunk, **_mel_cfg_kwargs(self.audio_cfg))
        tensor = melspec_to_tensor(mel)
        return {"spectrogram": tensor, "row_id": row_id}
