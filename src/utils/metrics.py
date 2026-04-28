"""
src/utils/metrics.py
─────────────────────
BirdCLEF+ 2026 공식 평가 지표:
  "보정된 macro-averaged ROC-AUC"
  — 참 양성(label=1)이 존재하는 클래스만 포함하여 평균
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_roc_auc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_positive: int = 1,
) -> float:
    """대회 공식 metric.

    Parameters
    ----------
    y_true       : (N_samples, N_classes)  — 이진 라벨
    y_pred       : (N_samples, N_classes)  — 예측 확률
    min_positive : 양성 샘플이 이 수 이상인 클래스만 AUC 계산에 포함

    Returns
    -------
    float : macro-averaged ROC-AUC (유효 클래스만)
    """
    n_classes = y_true.shape[1]
    aucs = []

    for c in range(n_classes):
        col_true = y_true[:, c]
        if col_true.sum() < min_positive:
            # 양성 샘플 없음 → 건너뜀
            continue
        try:
            auc = roc_auc_score(col_true, y_pred[:, c])
            aucs.append(auc)
        except ValueError:
            pass  # 단일 클래스 상황 무시

    return float(np.mean(aucs)) if aucs else 0.0


def print_per_class_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_k: int = 20,
) -> None:
    """상위/하위 k개 클래스 AUC 출력 (디버깅용)."""
    results = []
    for c, name in enumerate(class_names):
        col = y_true[:, c]
        if col.sum() == 0:
            continue
        try:
            auc = roc_auc_score(col, y_pred[:, c])
            results.append((name, auc, int(col.sum())))
        except ValueError:
            pass

    results.sort(key=lambda x: x[1], reverse=True)
    header = f"{'Class':<30} {'AUC':>6} {'#Pos':>6}"
    print(header)
    print("-" * len(header))
    for row in results[:top_k]:
        print(f"{row[0]:<30} {row[1]:.4f} {row[2]:>6}")
    print("  ...")
    for row in results[-top_k:]:
        print(f"{row[0]:<30} {row[1]:.4f} {row[2]:>6}")
    macro = np.mean([r[1] for r in results])
    print(f"\n[Macro AUC (valid classes only)] {macro:.4f}  ({len(results)} classes)")
