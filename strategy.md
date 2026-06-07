# BirdCLEF+ 2026 전략 정리 (notebook 기준)

## 목적
`/notebooks/birdclef-2026-visual-cpu-inference.ipynb`를 크게 흔들지 않고, 점수 개선 가능성이 높은 항목만 우선 적용한다.

## 현재 노트북에서 이미 구현된 것
- Perch + ProtoSSM + ResidualSSM 시계열 모델링
- Site/Hour prior 및 후처리 스무딩
- MLP probes (embedding + sequential feature)
- Isotonic 기반 class-wise calibration/threshold
- SED/BirdNET 분기 + rank blending

즉, 아래 항목은 “미구현”이 아니라 “이미 부분 구현” 상태다.
- class-wise threshold optimization
- calibration
- temporal modeling

## 미구현/보강 우선순위 (ROI 기준)
1. Test-time adaptation (보수적 pseudo-label)
2. Hard negative mining (클래스별 상한)
3. `train_audio` + secondary label soft target (사전 캐시 방식)

## 최소 변경 적용안 (현재 코드 반영)
Cell 11에서 `train_mlp_probes()` 직전에, pseudo-label을 MLP probe 학습 데이터에만 제한적으로 추가한다.

핵심 규칙 (v2):
- 사용 위치: Cell 11, `apply_prior(sc_te)` 이후
- 조건: `top1 >= 0.992` and `(top1 - top2) >= 0.72`
- 라벨: selected window에 대해 top1 클래스만 one-hot pseudo positive
- 상한: 전체 최대 700 rows, 클래스별 최대 35 rows
- fallback: 선택 샘플이 0개면 기존 학습 경로 유지

의도:
- ProtoSSM/ResidualSSM 본체는 건드리지 않는다.
- MLP probe만 test 분포에 약하게 적응시켜 domain gap 완화 효과를 노린다.

## 추가 적용 (2번/3번)
- 2번(튜닝): pseudo-label preset을 `safe_v2`로 적용
- 3번(hard negative): Cell 08 `train_mlp_probes()`에 클래스별 hard negative oversampling 추가
- hard negative 기본값:
`hard_neg_thr=0.80`, `hard_neg_max=240`, `hard_neg_repeat=2`

## 주의사항
- transductive/pseudo-label 방식은 대회 규정 위반 소지가 없는지 제출 전 반드시 확인한다.
- pseudo-label은 노이즈 유입 위험이 있으므로, threshold를 완화하기보다 보수적으로 유지한다.

## 다음 실험 순서
1. `safe_v2 + hard negative` 조합으로 LB 재측정
2. pseudo-label 선택량(로그 출력)과 hard-neg rows를 함께 기록해 상관관계 확인
3. 필요 시 `hard_neg_thr`(0.80→0.82) 또는 `hard_neg_repeat`(2→1)로 미세 완화
