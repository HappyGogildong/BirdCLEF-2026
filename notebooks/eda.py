"""
notebooks/eda.py
─────────────────
BirdCLEF 2026 탐색적 데이터 분석 (EDA).

Kaggle Notebook / Jupyter 에서 셀 단위로 실행하거나
python notebooks/eda.py 로 전체 실행 가능.

분석 항목
──────────
1. 클래스(종) 분포 — long-tail 확인
2. 동물군(taxa) 별 샘플 수
3. 지리적 분포 (위·경도)
4. 녹음 길이 분포
5. Soundscape 라벨 분포 (nocall 비율)
6. 멜-스펙트로그램 샘플 시각화
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# ── 경로 설정 (로컬 실행 시 수정) ────────────────────────────────────────────
#kaggle notebook path
#DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
DATA_ROOT = Path("/kaggle/birdclef-2026")
TRAIN_META = DATA_ROOT / "train.csv"
TRAIN_LABELS = DATA_ROOT / "train_soundscape_labels.csv"
TRAIN_AUDIO = DATA_ROOT / "train_audio"
SOUNDSCAPES = DATA_ROOT / "train_soundscapes"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
TAXONOMY = DATA_ROOT / "taxonomy.csv"
LOCATIONS = DATA_ROOT / "recording_location.txt"

print("=" * 60)
print("  BirdCLEF+ 2026 — EDA")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1: 메타데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] 메타데이터 로드")
meta = pd.read_csv(TRAIN_META)
print(f"  shape: {meta.shape}")
print(meta.head(3).to_string())
print("\n  dtypes:")
print(meta.dtypes)
print("\n  결측치 수:")
print(meta.isnull().sum()[meta.isnull().sum() > 0])

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2: 클래스 분포 (종별 샘플 수)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] 클래스 분포")
class_counts = meta["primary_label"].value_counts()
print(f"  총 종 수     : {len(class_counts)}")
print(f"  최다 샘플 종 : {class_counts.index[0]} ({class_counts.iloc[0]}개)")
print(f"  최소 샘플 종 : {class_counts.index[-1]} ({class_counts.iloc[-1]}개)")
print(f"  중앙값       : {class_counts.median():.0f}개")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 상위 40개
top40 = class_counts.head(40)
axes[0].bar(range(len(top40)), top40.values, color="steelblue")
axes[0].set_xticks(range(len(top40)))
axes[0].set_xticklabels(top40.index, rotation=90, fontsize=7)
axes[0].set_title("상위 40 클래스 샘플 수")
axes[0].set_ylabel("샘플 수")

# 분포 히스토그램
axes[1].hist(class_counts.values, bins=40, color="coral", edgecolor="white")
axes[1].set_title("클래스별 샘플 수 분포 (Long-tail)")
axes[1].set_xlabel("샘플 수")
axes[1].set_ylabel("클래스 수")
axes[1].axvline(class_counts.median(), color="navy", linestyle="--", label=f"중앙값={class_counts.median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("class_distribution.png", bbox_inches="tight")
plt.show()
print("  → class_distribution.png 저장")

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3: 지리적 분포
# ─────────────────────────────────────────────────────────────────────────────
if {"latitude", "longitude"}.issubset(meta.columns):
    print("\n[3] 지리적 분포")
    geo = meta.dropna(subset=["latitude", "longitude"])
    print(f"  좌표 있는 샘플: {len(geo):,} / {len(meta):,}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        geo["longitude"], geo["latitude"],
        c=pd.Categorical(geo["primary_label"]).codes,
        cmap="tab20", alpha=0.3, s=5
    )
    ax.set_title("녹음 위치 분포 (종 코드별 색상)")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    plt.tight_layout()
    plt.savefig("geo_distribution.png", bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4: Soundscape 라벨 분포
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Soundscape 라벨 분포")
labels_df = pd.read_csv(TRAIN_LABELS)
print(f"  shape: {labels_df.shape}")
print(labels_df.head(5).to_string())

nocall_ratio = (labels_df["species"] == "nocall").mean()
print(f"\n  nocall 비율     : {nocall_ratio:.1%}")
print(f"  유효 라벨 비율  : {1 - nocall_ratio:.1%}")

# 구간당 종 수 분포
labels_df["n_species"] = labels_df["species"].apply(
    lambda s: 0 if str(s).lower() == "nocall" else len(str(s).split())
)
fig, ax = plt.subplots(figsize=(8, 4))
labels_df["n_species"].value_counts().sort_index().plot(kind="bar", ax=ax, color="teal")
ax.set_title("5초 구간당 동시 출현 종 수")
ax.set_xlabel("출현 종 수")
ax.set_ylabel("구간 수")
plt.tight_layout()
plt.savefig("cooccurrence_distribution.png", bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5: 멜-스펙트로그램 시각화
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] 멜-스펙트로그램 샘플 시각화")
import librosa
import librosa.display

sample_files = list(TRAIN_AUDIO.glob("**/*.ogg"))[:6]
if sample_files:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.flatten(), sample_files):
        y, sr = librosa.load(str(f), sr=32000, mono=True, duration=5.0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=50, fmax=14000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=320, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f.parent.name, fontsize=9)
        ax.set_xlabel("")
    plt.suptitle("멜-스펙트로그램 샘플 (6종)", fontsize=13)
    plt.tight_layout()
    plt.savefig("melspec_samples.png", bbox_inches="tight")
    plt.show()
else:
    print("  train_audio 파일 없음 (Kaggle 환경에서 실행하세요)")

# ─────────────────────────────────────────────────────────────────────────────
# 섹션 6: sample_submission 구조 확인
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] sample_submission 구조")
sub = pd.read_csv(SAMPLE_SUB)
print(f"  shape: {sub.shape}")
print(f"  컬럼 예시: {list(sub.columns[:5])} ... {list(sub.columns[-3:])}")
print(f"  row_id 예시: {sub['row_id'].head(3).tolist()}")

print("\n✓ EDA 완료.")
