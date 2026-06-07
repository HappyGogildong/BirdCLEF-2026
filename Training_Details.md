# BirdCLEF 2026: 학습 방식 상세 문서

이 문서는 BirdCLEF 2026 조류 음향 식별(다중 레이블 분류) 대회 모델의 학습 방법론, 네트워크 아키텍처, 손실 함수 및 최적화 전략을 상술합니다.

## 1. 학습 방식 및 파이프라인
오디오 파형(Waveform)을 5초 단위로 잘라 전처리한 뒤 1채널 멜-스펙트로그램(Mel-Spectrogram) 이미지로 변환하여 2D CNN 기반 영상 분류 모델로 학습하는 **도메인 전이(Domain Transfer) 학습 방식**을 채택하고 있습니다. 한 구간(5초)에 여러 마리의 새가 동시에 지저귈 수 있으므로 **다중 레이블 분류(Multi-Label Classification)**로 접근합니다.

## 2. 학습 모델 (Backbone)
* **기본 모델**: `tf_efficientnet_b2_ns` (timm 라이브러리 활용)
* **선택 이유**: 
  * ImageNet에서 100만 장 이상의 데이터로 Noisy-Student 기법이 적용되어 사전 학습된(Pre-trained) 모델입니다.
  * 파라미터 수가 약 9.1M으로 가벼워 Kaggle 환경의 제한된 추론 시간(9시간)을 통과하기 매우 적합하며, 과거 다수의 오디오 분류 대회(BirdCLEF 등) 우승 솔루션에서 강력한 성능이 검증된 베이스라인입니다.

## 3. 모델 구조 변경 사항 (Architecture Modifications)

### 3.1. 단일 채널(1-Channel) 입력 처리
* **문제점**: ImageNet 사전학습 모델은 RGB 3채널 이미지를 입력으로 받지만, 생성된 멜-스펙트로그램은 1채널(Grayscale)입니다.
* **해결책**: 첫 번째 Convolution 레이어의 3채널 가중치(Weight) 값을 채널 차원 축으로 평균(`mean`) 내어 1채널 가중치로 압축합니다. 이를 통해 사전학습된 공간적 필터 특성을 보존하면서 단일 채널 스펙트로그램을 그대로 입력받을 수 있습니다 (`src/models/backbone.py` 참조).

### 3.2. Classifier (Head) 구조 및 구현 방식
* **GeM Pooling (Generalized Mean Pooling)**:
  * timm의 기본 Global Average Pooling 대신 **GeM Pooling (p=3.0)**을 사용합니다.
  * p=1이면 Average Pooling, p=∞이면 Max Pooling과 같으며, p=3은 시각 및 음향 태스크에서 배경음과 실제 타겟 소리를 구별하는 데 경험적으로 뛰어난 성능을 보입니다. 또한 SpecAugment로 마스킹된 영역에 덜 민감하게 동작하는 장점이 있습니다.
* **Head 레이어 구성**:
  * Backbone Feature (B, 1408) → GeM Pooling → `Flatten`
  * `BatchNorm1d(1408)` → `Dropout(0.3)`
  * `Linear(1408 → 512)` → `GELU` 활성화 함수
  * `BatchNorm1d(512)` → `Dropout(0.15)`
  * `Linear(512 → 234)` (최종 234개 조류 클래스 예측)

## 4. 학습 최적화 및 LR 스케줄링 (Optimizer & Scheduler)
* **Optimizer**: `AdamW`
  * Learning Rate: `1.0e-4`
  * Weight Decay: `1.0e-2`로 설정해 정규화(Regularization) 효과 부여.
* **Scheduler**: `Cosine Annealing with Warmup`
  * **Warmup Epochs (2 Epoch)**: 학습 초기에 LR을 천천히 0에서 목표치(1.0e-4)까지 끌어올려, 초기 랜덤 가중치 상태에서 Loss가 발산하거나 그래디언트가 폭발하는 현상을 방지합니다.
  * **Cosine Decay (T_max=30)**: Warmup 이후 코사인 곡선을 따라 점진적으로 최저 LR(`1.0e-6`)까지 부드럽게 감소시킵니다. 현재 기본 설정인 30 Epoch에 맞춰 완벽히 수렴하도록 디자인되었습니다.

## 5. 손실 함수 (Loss Function)
극심한 클래스 불균형(Class Imbalance)과 다중 클래스 환경을 극복하기 위해 단순히 `BCEWithLogitsLoss`를 넘어 고도화된 커스텀 손실 함수를 사용합니다.

### 5.1. Focal BCELoss (기본)
* **Focal Term (gamma=2.0)**: 새소리가 없는 `nocall` 구간이나 매우 쉽게 맞출 수 있는 샘플이 Loss에 기여하는 비중을 대폭 줄이고, 맞추기 어려운(확률이 낮은) 소수 클래스 샘플에 가중치를 집중합니다.
* **Positional Weight (`pos_weight`)**: 데이터셋에 등장하는 빈도의 역수를 클래스 가중치로 환산하여, 소수 종(Rare species)을 틀렸을 때의 패널티를 강화합니다. 최대 가중치는 10배(`pos_weight_clip=10.0`)로 제한합니다.
* **Label Smoothing (0.05)**: 타겟 라벨을 1과 0이 아닌 `0.95`, `0.05` 등 Soft-label로 변환하여 모델이 특정 타겟에 과도하게 확신(Over-confidence)을 가지는 현상을 억제합니다.

### 5.2. Asymmetric Loss (대안)
* 옵션으로 적용 가능한 이 Loss는 양성(Target)과 음성(Non-target)의 감마 값을 다르게 줍니다.
* 새가 존재하지 않는 텅 빈 구간(Negative)에 대해서는 강한 패널티(gamma_neg=4.0)로 Loss를 깎아내고, 새가 존재하는 구간(Positive)에 대해서는 있는 그대로(gamma_pos=0.0) 평가하여, False Positive(오탐지)를 억제하는 데 특화된 Loss입니다.
