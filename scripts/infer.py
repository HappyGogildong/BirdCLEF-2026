"""
scripts/infer.py
─────────────────
BirdCLEF 2026 추론 + 제출 파일 생성 진입점.

사용 예
────────
# 전체 fold 앙상블 추론
python scripts/infer.py

# 특정 fold만 사용
python scripts/infer.py inference.tta_folds=[0,1,2]

# TTA 비활성화
python scripts/infer.py inference.tta=false
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import build_label_map, load_metadata
from src.inference.predict import ensemble_predict, save_submission


def main():
    # ── 설정 로드 ──────────────────────────────────────────────────────────────
    base_cfg = OmegaConf.load(
        Path(__file__).parent.parent / "configs" / "config.yaml"
    )
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 클래스 정보 로드 ───────────────────────────────────────────────────────
    meta_df = load_metadata(cfg.paths.train_meta)
    sub_df = pd.read_csv(cfg.paths.sample_submission)
    label2idx, class_names = build_label_map(meta_df, sub_df)

    fold_ids = list(cfg.inference.tta_folds)
    print(f"앙상블 fold: {fold_ids}  |  TTA: {cfg.inference.tta}")

    # ── 앙상블 추론 ───────────────────────────────────────────────────────────
    pred_df = ensemble_predict(
        cfg=cfg,
        submission_df=sub_df,
        class_names=class_names,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        device=device,
        fold_ids=fold_ids,
    )

    # ── 후처리: 앞뒤 1칸씩(총 3칸, 15초) 이동 평균 적용 (Sequence-aware smoothing) ──
    # row_id는 'audio_id_endsec' 형식이므로 마지막 '_'를 기준으로 audio_id를 추출합니다.
    pred_df['audio_id'] = pred_df['row_id'].apply(lambda x: x.rsplit('_', 1)[0])
    
    # 타겟 확률 컬럼만 선택하여 스무딩
    class_cols = [c for c in pred_df.columns if c not in ['row_id', 'audio_id']]
    smoothed_probs = pred_df.groupby('audio_id')[class_cols].rolling(window=3, center=True, min_periods=1).mean().reset_index(0, drop=True)
    
    # 덮어쓰기 및 임시 컬럼 삭제
    pred_df[class_cols] = smoothed_probs
    pred_df = pred_df.drop(columns=['audio_id'])

    # ── 제출 파일 저장 ─────────────────────────────────────────────────────────
    output_path = Path(cfg.paths.output_dir) / "submission.csv"
    save_submission(pred_df, output_path)


if __name__ == "__main__":
    main()
