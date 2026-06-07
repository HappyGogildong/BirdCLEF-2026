"""
scripts/train.py
─────────────────
BirdCLEF 2026 학습 진입점.

사용 예
────────
# 단일 fold
python scripts/train.py training.fold=0

# 전체 fold 순차 실행
for fold in 0 1 2 3 4; do
    python scripts/train.py training.fold=$fold
done

# 손실 함수 변경
python scripts/train.py training.loss.name=asymmetric training.fold=0
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

from src.data.augmentation import TrainAugmentation, WaveformMixup
from src.data.dataset import BirdClefTrainDataset, SoundscapeDataset
from src.data.preprocessing import (
    build_label_map,
    compute_class_weights,
    load_metadata,
    load_soundscape_labels,
    split_kfold,
)
from src.models.classifier import BirdClefClassifier
from src.training.loss import build_loss
from src.training.scheduler import build_optimizer, build_scheduler
from src.training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────────
# 시드 고정
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# 샘플러: 클래스 불균형 완화 (오버샘플링)
# ──────────────────────────────────────────────────────────────────────────────

def build_combined_sampler(train_df, soundscape_df, label2idx: dict) -> WeightedRandomSampler:
    """train_audio와 train_soundscapes를 결합한 데이터셋용 역빈도 가중치 샘플러."""
    counts = train_df["primary_label"].value_counts().to_dict()
    
    # 1. train_audio 샘플 가중치
    train_weights = [1.0 / counts.get(row["primary_label"], 1.0) for _, row in train_df.iterrows()]
    
    # 2. soundscape 샘플 가중치 (train 가중치 평균의 1.5배를 부여하여 도메인 적응 강화)
    if train_weights:
        avg_weight = sum(train_weights) / len(train_weights)
        soundscape_weight = avg_weight * 1.5
    else:
        soundscape_weight = 1.0
    soundscape_weights = [soundscape_weight] * len(soundscape_df)
    
    # 3. 두 가중치 결합
    all_weights = train_weights + soundscape_weights
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(all_weights, dtype=torch.float64),
        num_samples=len(all_weights),
        replacement=True,
    )
    return sampler


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 설정 로드 ──────────────────────────────────────────────────────────────
    base_cfg = OmegaConf.load(
        Path(__file__).parent.parent / "configs" / "config.yaml"
    )
    # CLI 오버라이드 지원 (ex: training.fold=1)
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    fold = cfg.training.fold
    set_seed(cfg.training.seed + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # ── 메타데이터 로드 ────────────────────────────────────────────────────────
    print("\n[1] 메타데이터 로드 중...")
    meta_df = load_metadata(cfg.paths.train_meta)
    soundscape_df = load_soundscape_labels(cfg.paths.train_labels)

    import pandas as pd
    sub_df = pd.read_csv(cfg.paths.sample_submission)
    label2idx, class_names = build_label_map(meta_df, sub_df)
    n_classes = len(class_names)
    print(f"  총 클래스 수: {n_classes}")
    print(f"  train_audio  샘플 수: {len(meta_df):,}")
    print(f"  soundscape   구간 수: {len(soundscape_df):,}")

    # ── Fold 분할 ──────────────────────────────────────────────────────────────
    print("\n[2] Fold 분할 중...")
    meta_df = split_kfold(meta_df, n_splits=cfg.training.n_folds, seed=cfg.training.seed)
    train_df = meta_df[meta_df["fold"] != fold].reset_index(drop=True)
    valid_df = meta_df[meta_df["fold"] == fold].reset_index(drop=True)
    print(f"  Fold {fold} → Train: {len(train_df):,} | Valid: {len(valid_df):,}")

    # ── 클래스 가중치 ──────────────────────────────────────────────────────────
    pos_weight = torch.tensor(
        compute_class_weights(meta_df, label2idx, cfg.training.loss.pos_weight_clip),
        device=device,
    )

    # ── Dataset / DataLoader ───────────────────────────────────────────────────
    print("\n[3] Dataset 구성 중...")
    mixup = WaveformMixup(
        alpha=cfg.augmentation.train.mixup.alpha,
        p=cfg.augmentation.train.mixup.p,
    )
    aug = TrainAugmentation(cfg.augmentation.train, mixup_helper=mixup)

    train_ds = BirdClefTrainDataset(
        df=train_df,
        audio_dir=cfg.paths.train_audio,
        label2idx=label2idx,
        audio_cfg=cfg.audio,
        is_train=True,
        augmentation=aug,
    )
    
    soundscape_ds = SoundscapeDataset(
        labels_df=soundscape_df,
        soundscape_dir=cfg.paths.train_soundscapes,
        label2idx=label2idx,
        audio_cfg=cfg.audio,
        is_train=True,
        augmentation=aug,
    )
    
    combined_train_ds = ConcatDataset([train_ds, soundscape_ds])

    valid_ds = BirdClefTrainDataset(
        df=valid_df,
        audio_dir=cfg.paths.train_audio,
        label2idx=label2idx,
        audio_cfg=cfg.audio,
        is_train=False,
    )

    sampler = build_combined_sampler(train_df, soundscape_df, label2idx)
    train_loader = DataLoader(
        combined_train_ds,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # ── 모델 / 손실 / 옵티마이저 ──────────────────────────────────────────────
    print("\n[4] 모델 초기화 중...")
    model = BirdClefClassifier(cfg.model)
    criterion = build_loss(cfg.training.loss, pos_weight=pos_weight)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler, cfg.training.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터 수: {total_params / 1e6:.1f}M")

    # ── WandB (선택) ───────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=f"fold{fold}_{cfg.model.backbone}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── 학습 ──────────────────────────────────────────────────────────────────
    print("\n[5] 학습 시작...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
        output_dir=cfg.paths.checkpoint_dir,
        fold=fold,
        wandb_run=wandb_run,
    )
    best_auc = trainer.fit(train_loader, valid_loader)
    print(f"\n완료! Best AUC (Fold {fold}): {best_auc:.4f}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
