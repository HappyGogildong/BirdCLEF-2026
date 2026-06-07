import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Kaggle 환경에서 'src' 폴더를 자동으로 찾아 sys.path에 추가합니다.
kaggle_input = Path('/kaggle/input')
src_found = False
config_path = None
checkpoint_dir_found = None

if kaggle_input.exists():
    # 모델 가중치 폴더 동적 탐색 (.pth 파일이 있는 디렉토리)
    for p in kaggle_input.rglob('*.pth'):
        checkpoint_dir_found = str(p.parent)
        print(f"[설정] 모델 체크포인트 경로를 찾았습니다: {checkpoint_dir_found}")
        break


    # /kaggle/input 아래에 있는 모든 'src' 디렉토리 탐색
    for p in kaggle_input.rglob('src'):
        if p.is_dir() and (p / 'inference' / 'predict.py').exists():
            parent_dir = str(p.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            print(f"[설정] '{parent_dir}' 경로를 sys.path에 추가했습니다.")
            src_found = True
            config_path = p.parent / 'configs' / 'config.yaml'
            break

# 로컬(현재 윈도우 환경)에서 테스트 시 대응 (노트북 셀 오류 방지를 위해 __file__ 대신 os.getcwd() 활용)
import os
if not src_found:
    LOCAL_SRC = Path(os.getcwd()).resolve()
    if (LOCAL_SRC / "src").exists() and str(LOCAL_SRC) not in sys.path:
        sys.path.insert(0, str(LOCAL_SRC))
    config_path = LOCAL_SRC / "configs" / "config.yaml"

from omegaconf import OmegaConf
from src.data.preprocessing import build_label_map, load_metadata
from src.inference.predict import ensemble_predict, save_submission

def main():
    ckpt_path_override = checkpoint_dir_found if checkpoint_dir_found else "/kaggle/input/test-birdclef-models/checkpoints"

    # Kaggle dataset 이름 (데이터셋 패널에 추가된 이름으로 변경 가능)
    DATASET_NAME = "test-birdclef-2026"
    dataset_root = f"/kaggle/input/{DATASET_NAME}"

    # 1. Kaggle 환경에 맞게 기본 경로 설정 오버라이드
    # Kaggle 데이터셋 경로는 일반적으로 /kaggle/input 아래에 마운트됩니다.
    kaggle_overrides = [
        f"paths.data_root={dataset_root}",
        f"paths.train_meta={dataset_root}/train.csv",
        f"paths.test_soundscapes={dataset_root}/test_soundscapes",
        f"paths.sample_submission={dataset_root}/sample_submission.csv",
        "paths.output_dir=/kaggle/working",  # 출력은 반드시 /kaggle/working 에 해야 제출 가능
        f"paths.checkpoint_dir={ckpt_path_override}", # 동적으로 찾은 모델 경로 할당

        "model.pretrained=false", # [중요] 오프라인 환경에서 timm 모델 다운로드 방지
        "inference.tta=true", # TTA 활성화
        "training.batch_size=16",
        "training.num_workers=2"
    ]

    # 로컬 테스트용 설정 (로컬에 /kaggle/input 가 없는 경우)
    if not Path("/kaggle/input").exists():
        print("로컬 환경으로 인식되었습니다.")
        kaggle_overrides = [
            "paths.output_dir=./outputs",
        ]

    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(kaggle_overrides)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. 메타데이터에서 클래스 이름 매핑 정보 로드
    # Kaggle의 test_soundscapes 에서는 클래스가 숨겨져 있지 않으므로 train.csv로 레이블맵 빌드 가능
    print(f"Loading metadata from {cfg.paths.train_meta}...")
    meta_df = load_metadata(cfg.paths.train_meta)
    
    print(f"Loading sample submission from {cfg.paths.sample_submission}...")
    sub_df = pd.read_csv(cfg.paths.sample_submission)
    
    label2idx, class_names = build_label_map(meta_df, sub_df)
    
    # 3. 앙상블에 사용할 fold 지정
    fold_ids = list(cfg.inference.tta_folds)
    print(f"Ensemble folds: {fold_ids} | TTA Enabled: {cfg.inference.tta}")

    # 4. 추론 수행 (src/inference/predict.py 의 로직 사용)
    # 각 fold의 모델을 로드하여 5초 구간 스펙트로그램에 대한 출력을 평균냅니다.
    print("Starting prediction...")
    pred_df = ensemble_predict(
        cfg=cfg,
        submission_df=sub_df,
        class_names=class_names,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        device=device,
        fold_ids=fold_ids,
    )

    # 5. 같은 audio_id 내에서 앞뒤 1칸씩(총 3칸, 15초) 이동 평균 적용 (Sequence-aware smoothing)
    # row_id는 'audio_id_endsec' 형식이므로 마지막 '_'를 기준으로 audio_id를 추출합니다.
    pred_df['audio_id'] = pred_df['row_id'].apply(lambda x: x.rsplit('_', 1)[0])
    
    # 타겟이 되는 종(확률) 컬럼만 선택하여 스무딩
    class_cols = [c for c in pred_df.columns if c not in ['row_id', 'audio_id']]
    smoothed_probs = pred_df.groupby('audio_id')[class_cols].rolling(window=3, center=True, min_periods=1).mean().reset_index(0, drop=True)
    
    # 원래 데이터프레임에 스무딩된 결과 덮어쓰기 및 임시 컬럼 삭제
    pred_df[class_cols] = smoothed_probs
    pred_df = pred_df.drop(columns=['audio_id'])

    # 6. submission.csv 저장
    # 캐글 규정에 따라 /kaggle/working/submission.csv 에 저장
    output_path = Path(cfg.paths.output_dir) / "submission.csv"
    save_submission(pred_df, output_path)
    print("Inference completed successfully. submission.csv generated.")

if __name__ == "__main__":
    main()
