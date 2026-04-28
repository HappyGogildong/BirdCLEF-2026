"""
src/data/preprocessing.py
──────────────────────────
메타데이터 CSV 로딩, 클래스 인덱스 매핑, StratifiedGroupKFold 분할.

주요 함수
─────────
load_metadata()         : train_metadata.csv → DataFrame
build_label_map()       : 종 코드 → 정수 인덱스 매핑 dict
split_kfold()           : StratifiedGroupKFold (그룹=audio_id, 층=primary_label)
load_soundscape_labels(): train_soundscape_labels.csv → 5초 단위 멀티핫 행렬
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ──────────────────────────────────────────────────────────────────────────────
# 메타데이터 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """train_metadata.csv를 로드하고 기본 전처리를 수행한다.

    주요 컬럼
    ---------
    filename        : 'species_code/XC12345.ogg' 형태
    primary_label   : 종 코드 (eBird 코드)
    secondary_labels: 보조 종 목록 (문자열; 파싱 필요)
    latitude, longitude
    date            : 'YYYY-MM-DD'
    """
    df = pd.read_csv(csv_path)

    # 파일 경로를 audio_id(줄기 이름) 컬럼으로 통일
    df["audio_id"] = df["filename"].apply(lambda x: Path(x).stem)

    # 날짜 파싱 (가능한 경우)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month

    # 좌표 이상치 제거
    if "latitude" in df.columns:
        df = df[(df["latitude"].between(-90, 90)) | df["latitude"].isna()]
    if "longitude" in df.columns:
        df = df[(df["longitude"].between(-180, 180)) | df["longitude"].isna()]

    df = df.reset_index(drop=True)
    return df


def load_soundscape_labels(csv_path: str | Path) -> pd.DataFrame:
    """train_soundscape_labels.csv 로드.

    컬럼: row_id, site, seconds, audio_id, species (공백 구분 종 목록)
    """
    df = pd.read_csv(csv_path)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 클래스 인덱스 매핑
# ──────────────────────────────────────────────────────────────────────────────

def build_label_map(
    metadata: pd.DataFrame,
    submission: pd.DataFrame | None = None,
) -> Tuple[Dict[str, int], List[str]]:
    """종 코드 → 정수 인덱스 매핑.

    Parameters
    ----------
    metadata   : train_metadata.csv DataFrame (primary_label 컬럼 포함)
    submission : sample_submission.csv (헤더 컬럼에서 직접 종 목록 파싱)

    Returns
    -------
    label2idx : dict[str, int]
    idx2label : list[str]
    """
    if submission is not None:
        # 제출 파일의 컬럼 = ['row_id', 'species_1', 'species_2', ...]
        classes = [c for c in submission.columns if c != "row_id"]
    else:
        classes = sorted(metadata["primary_label"].dropna().unique().tolist())

    label2idx = {cls: i for i, cls in enumerate(classes)}
    return label2idx, classes


def encode_labels(
    species_str: str,
    label2idx: Dict[str, int],
    n_classes: int,
) -> np.ndarray:
    """공백으로 구분된 종 문자열 → 멀티핫 벡터 (n_classes,).

    'nocall' 또는 빈 문자열이면 모두 0인 벡터 반환.
    """
    vec = np.zeros(n_classes, dtype=np.float32)
    if not isinstance(species_str, str) or species_str.strip().lower() == "nocall":
        return vec
    for sp in species_str.strip().split():
        idx = label2idx.get(sp)
        if idx is not None:
            vec[idx] = 1.0
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# K-Fold 분할
# ──────────────────────────────────────────────────────────────────────────────

def split_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """StratifiedGroupKFold로 fold 컬럼을 추가한다.

    층(stratify) : primary_label  → 종별 분포 유지
    그룹(group)  : audio_id       → 동일 녹음이 train/val에 동시 존재하지 않도록
    """
    df = df.copy()
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        skf.split(df, df["primary_label"], groups=df["audio_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 클래스 불균형 가중치 계산
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    metadata: pd.DataFrame,
    label2idx: Dict[str, int],
    clip_max: float = 10.0,
) -> np.ndarray:
    """Inverse-frequency 방식의 pos_weight (BCEWithLogitsLoss 용).

    pos_weight[c] = (N_neg / N_pos).clip(max=clip_max)
    """
    n_classes = len(label2idx)
    n_total = len(metadata)
    counts = np.zeros(n_classes, dtype=np.float64)

    for sp, idx in label2idx.items():
        counts[idx] = (metadata["primary_label"] == sp).sum()

    pos = np.maximum(counts, 1.0)
    neg = n_total - pos
    weights = np.clip(neg / pos, a_min=1.0, a_max=clip_max)
    return weights.astype(np.float32)
