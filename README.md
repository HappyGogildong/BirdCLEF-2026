# BirdCLEF+ 2026 — 솔루션 파이프라인

브라질 파나마(Pantanal) 습지 야생동물 음향 자동 식별 대회.  
1분 녹음을 **5초 단위**로 분할해 **234종**의 출현 확률을 예측한다.  
평가 지표: **Calibrated macro-averaged ROC-AUC** (양성 클래스 존재 종만)

---

## 디렉터리 구조

```
birdclef2026/
├── requirements.txt              # 의존 라이브러리
├── configs/
│   └── config.yaml               # 전체 실험 설정 (경로/오디오/모델/학습/증강/추론)
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # 메타데이터 로드, KFold 분할, 클래스 가중치
│   │   ├── dataset.py            # PyTorch Dataset (train/soundscape/test)
│   │   └── augmentation.py       # 파형·스펙트로그램 증강 (Mixup, SpecAugment)
│   ├── models/
│   │   ├── backbone.py           # timm EfficientNet 래퍼 (1ch 변환 포함)
│   │   └── classifier.py         # GeM Pooling + BN/Dropout Head
│   ├── training/
│   │   ├── loss.py               # FocalBCE / AsymmetricLoss / factory
│   │   ├── trainer.py            # AMP + GradAccum + 체크포인트 저장
│   │   └── scheduler.py          # AdamW + Warmup-Cosine LR 스케줄러
│   ├── inference/
│   │   └── predict.py            # TTA + fold 앙상블 + submission 생성
│   └── utils/
│       ├── audio.py              # 파형 로드/정규화/멜스펙 변환
│       └── metrics.py            # macro-ROC-AUC (대회 공식 지표)
├── scripts/
│   ├── train.py                  # 학습 진입점
│   └── infer.py                  # 추론 진입점
└── notebooks/
    └── eda.py                    # 탐색적 데이터 분석
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행 순서

### 1. EDA
```bash
python notebooks/eda.py
```

### 2. 학습 (5-Fold)
```bash
# Fold 0 학습
python scripts/train.py training.fold=0

# 전체 Fold 순차 실행 (bash)
for fold in 0 1 2 3 4; do
    python scripts/train.py training.fold=$fold
done
```

### 3. 추론 & 제출 파일 생성
```bash
python scripts/infer.py
# → outputs/submission.csv 생성
```

---

## 모델 전략

| 항목 | 선택 | 근거 |
|------|------|------|
| **Backbone** | EfficientNet-B2-NS (Noisy-Student) | BirdCLEF 과거 우승 검증, ~9M 파라미터로 속도·성능 균형 |
| **Pooling** | GeM (p=3, 학습 가능) | SpecAugment 마스킹 영역에 덜 민감 |
| **Loss** | FocalBCE (γ=2) + pos_weight | 클래스 극심 불균형 + nocall 비율 높음 |
| **Optimizer** | AdamW + layer-wise LR decay | backbone 0.1× lr, head 1× lr |
| **Scheduler** | Linear Warmup (2 epoch) + CosineAnnealing | 안정적 수렴 |
| **증강** | Mixup + GaussianNoise + SpecAugment | 소수 클래스 일반화 |
| **추론** | 5-fold 앙상블 + TTA (flip×2) | 분산 감소 |

---

## 주요 하이퍼파라미터

```yaml
audio.sample_rate:      32000
audio.n_mels:           128
audio.clip_duration:    5.0   # 초

training.batch_size:    32
training.epochs:        30
training.optimizer.lr:  1e-4
training.loss.name:     bce_focal
training.loss.focal_gamma: 2.0
training.n_folds:       5
```

설정 변경은 `configs/config.yaml` 수정 또는 CLI 오버라이드:
```bash
python scripts/train.py training.loss.name=asymmetric model.backbone=convnext_small
```

---

## 대회 규칙 요약

- **제출 형식**: `row_id` + 234개 종 확률 컬럼 CSV
- **평가 지표**: Calibrated macro-averaged ROC-AUC
- **팀 규모**: 최대 5명
- **일 제출 한도**: 5회
- **상금**: 총 $50,000
