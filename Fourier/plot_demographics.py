# -*- coding: utf-8 -*-
"""
plot_demographics.py
====================
WFDB 레코드(.hea 헤더)에서 성별/나이를 모아 pandas DataFrame으로 만들고,
matplotlib 히스토그램으로 분포도를 그린다.

데이터셋: Chapman-Shaoxing 12-lead ECG Database (PhysioNet, WFDB 포맷)
각 .hea 파일에는 다음과 같은 주석 줄이 들어있다:
    #Age: 26
    #Sex: Male

사용법:
    python plot_demographics.py
"""

import os
import glob

import pandas as pd
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# 1. 설정 — 데이터 위치
# ──────────────────────────────────────────────────────────────────────
DATASET_DIR = "/Users/rightnwise/Korea/" \
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
WFDB_DIR = os.path.join(DATASET_DIR, "WFDBRecords")


# ──────────────────────────────────────────────────────────────────────
# 2. 모든 .hea 파일에서 나이/성별 읽기
# ──────────────────────────────────────────────────────────────────────
def parse_header(hea_path):
    """하나의 .hea 파일에서 (age, sex)를 뽑아낸다. 없으면 None."""
    age, sex = None, None
    with open(hea_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#Age:"):
                value = line.split(":", 1)[1].strip()
                # 'Unknown' 같은 비숫자 값은 NaN 처리
                age = int(value) if value.isdigit() else None
            elif line.startswith("#Sex:"):
                sex = line.split(":", 1)[1].strip()  # 'Male' / 'Female'
    return age, sex


def collect_demographics(wfdb_dir):
    """WFDBRecords 하위의 모든 .hea를 훑어 DataFrame을 만든다."""
    pattern = os.path.join(wfdb_dir, "**", "*.hea")
    records = []
    for hea_path in glob.glob(pattern, recursive=True):
        age, sex = parse_header(hea_path)
        records.append({
            "record": os.path.splitext(os.path.basename(hea_path))[0],
            "age": age,
            "sex": sex,
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────
# 3. 메인 — 데이터 수집 → 요약 출력 → 히스토그램
# ──────────────────────────────────────────────────────────────────────
def main():
    print("📂 .hea 파일을 스캔하는 중...")
    df = collect_demographics(WFDB_DIR)
    print(f"총 {len(df)}개 레코드를 읽었습니다.\n")

    # 간단한 요약
    print("성별 분포:")
    print(df["sex"].value_counts(dropna=False), "\n")
    print("나이 요약 통계:")
    print(df["age"].describe(), "\n")

    # 그림: 왼쪽=성별, 오른쪽=나이
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (1) 성별 분포 — 막대 형태의 히스토그램
    sex_counts = df["sex"].value_counts()
    axes[0].bar(sex_counts.index, sex_counts.values,
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Sex Distribution")
    axes[0].set_xlabel("Sex")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(sex_counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    # (2) 나이 분포 — 성별로 색을 나눠 겹쳐 그린 히스토그램
    ages_male = df.loc[df["sex"] == "Male", "age"].dropna()
    ages_female = df.loc[df["sex"] == "Female", "age"].dropna()
    bins = range(0, 101, 5)  # 5살 단위 구간
    axes[1].hist(ages_male, bins=bins, alpha=0.6,
                 label="Male", color="#4C72B0")
    axes[1].hist(ages_female, bins=bins, alpha=0.6,
                 label="Female", color="#DD8452")
    axes[1].set_title("Age Distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("WFDB Records — Demographics", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "demographics.png")
    fig.savefig(out_path, dpi=120)
    print(f"✅ 그래프를 저장했습니다: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
