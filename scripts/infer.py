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

    # ── 제출 파일 저장 ─────────────────────────────────────────────────────────
    output_path = Path(cfg.paths.output_dir) / "submission.csv"
    save_submission(pred_df, output_path)


if __name__ == "__main__":
    main()
