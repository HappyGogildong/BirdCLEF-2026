"""
src/inference/predict.py
─────────────────────────
테스트 추론 + TTA(Test Time Augmentation) + 앙상블.

추론 전략
──────────
1. 각 fold 의 best checkpoint 로드 → sigmoid 확률 평균 앙상블
2. TTA: 원본 + 시간 반전(Time Flip) + 주파수 마스킹(약한 SpecAugment) 평균
3. row_id 별 확률을 sample_submission.csv 형식으로 저장
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TestSoundscapeDataset
from src.models.classifier import BirdClefClassifier


# ──────────────────────────────────────────────────────────────────────────────
# TTA 변환
# ──────────────────────────────────────────────────────────────────────────────

def _tta_transforms(spec: torch.Tensor) -> List[torch.Tensor]:
    """spec: (B, 1, H, W)
    반환: 여러 증강 버전 리스트
    """
    variants = [spec]
    # 시간 축 좌우 반전
    variants.append(spec.flip(dims=[-1]))
    # 주파수 축 상하 반전 (드물게 유효)
    variants.append(spec.flip(dims=[-2]))
    return variants


# ──────────────────────────────────────────────────────────────────────────────
# 단일 모델 추론
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> Dict[str, np.ndarray]:
    """row_id → 확률 벡터 dict 반환."""
    model.eval()
    results: Dict[str, np.ndarray] = {}

    for batch in tqdm(loader, desc="  Inference", leave=False):
        specs = batch["spectrogram"].to(device)
        row_ids = batch["row_id"]

        with autocast(enabled=(device.type == "cuda")):
            if use_tta:
                probs_list = []
                for aug_spec in _tta_transforms(specs):
                    logits = model(aug_spec)
                    probs_list.append(torch.sigmoid(logits).cpu().numpy())
                probs = np.mean(probs_list, axis=0)
            else:
                logits = model(specs)
                probs = torch.sigmoid(logits).cpu().numpy()

        for rid, prob in zip(row_ids, probs):
            results[rid] = prob

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 앙상블 추론 (여러 fold)
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_predict(
    cfg,
    submission_df: pd.DataFrame,
    class_names: List[str],
    checkpoint_dir: str | Path,
    device: torch.device,
    fold_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """fold 별 체크포인트를 로드하고 앙상블 평균으로 submission을 생성.

    Parameters
    ----------
    cfg            : OmegaConf 전체 설정
    submission_df  : sample_submission.csv
    class_names    : idx2label 리스트
    checkpoint_dir : fold{k}_best.pth 가 있는 디렉터리
    device         : 추론 디바이스
    fold_ids       : 사용할 fold ID 리스트 (None=전체)

    Returns
    -------
    submission_df  : row_id + 종별 확률이 채워진 DataFrame
    """
    checkpoint_dir = Path(checkpoint_dir)
    fold_ids = fold_ids or list(range(cfg.training.n_folds))

    # 테스트 데이터셋 / 로더 공유
    test_ds = TestSoundscapeDataset(
        submission_df=submission_df,
        soundscape_dir=cfg.paths.test_soundscapes,
        audio_cfg=cfg.audio,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
        pin_memory=cfg.training.pin_memory,
    )

    # 전체 row_id 목록 & 누적 확률
    all_row_ids = submission_df["row_id"].tolist()
    n_classes = len(class_names)
    accum = np.zeros((len(all_row_ids), n_classes), dtype=np.float64)
    row_id_to_idx = {rid: i for i, rid in enumerate(all_row_ids)}

    for fold in fold_ids:
        ckpt_path = checkpoint_dir / f"fold{fold}_best.pth"
        if not ckpt_path.exists():
            print(f"[!] 체크포인트 없음: {ckpt_path} — 건너뜀")
            continue

        print(f"\n[Fold {fold}] 체크포인트 로드: {ckpt_path}")
        model = BirdClefClassifier(cfg.model).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

        fold_results = predict_single_model(
            model, test_loader, device, use_tta=cfg.inference.tta
        )

        for rid, prob in fold_results.items():
            idx = row_id_to_idx.get(rid)
            if idx is not None:
                accum[idx] += prob

    n_folds_used = len(fold_ids)
    avg_probs = accum / max(n_folds_used, 1)

    # DataFrame 생성
    pred_df = pd.DataFrame(avg_probs, columns=class_names)
    pred_df.insert(0, "row_id", all_row_ids)
    return pred_df


# ──────────────────────────────────────────────────────────────────────────────
# 제출 파일 저장
# ──────────────────────────────────────────────────────────────────────────────

def save_submission(pred_df: pd.DataFrame, output_path: str | Path) -> None:
    """submission CSV 저장 및 간단한 통계 출력."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    n_rows = len(pred_df)
    n_cols = len(pred_df.columns) - 1  # row_id 제외
    print(f"\n[제출 파일 저장] {output_path}")
    print(f"  행 수 : {n_rows:,}  (5초 구간)")
    print(f"  클래스: {n_cols}")
    print(f"  확률 범위: [{pred_df.iloc[:, 1:].values.min():.4f}, {pred_df.iloc[:, 1:].values.max():.4f}]")
