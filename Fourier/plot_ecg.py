# -*- coding: utf-8 -*-
"""
plot_ecg.py
===========
12-lead ECG(심전도) 파형을 그래프로 그려서 보여주는 코드.

데이터셋: Chapman-Shaoxing 12-lead ECG Database (PhysioNet, WFDB 포맷)
각 환자는 .hea(헤더) + .mat(신호) 두 파일 쌍으로 저장되어 있다.

사용법:
    python plot_ecg.py                 # 기본 샘플(JS00001) 그리기
    python plot_ecg.py JS00007         # 특정 레코드 이름으로 그리기
    python plot_ecg.py 01/010/JS00002  # 폴더 경로까지 직접 지정
"""

import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import wfdb  # PhysioNet WFDB 포맷 전용 라이브러리


# ──────────────────────────────────────────────────────────────────────
# 1. 설정 (CONFIG) — 데이터가 어디 있는지, 진단 사전이 어디 있는지
# ──────────────────────────────────────────────────────────────────────
# 데이터셋 최상위 폴더 (절대경로). 본인 환경에 맞게 한 곳만 바꾸면 된다.
DATASET_DIR = "/Users/rightnwise/Desktop/" \
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"

WFDB_DIR = os.path.join(DATASET_DIR, "WFDBRecords")
COND_CSV = os.path.join(DATASET_DIR, "ConditionNames_SNOMED-CT.csv")


# ──────────────────────────────────────────────────────────────────────
# 2. 진단 코드 사전 만들기
#    .hea 파일의 #Dx 에는 SNOMED-CT 숫자코드가 들어있다.
#    예: 164889003 -> "AFIB (Atrial Fibrillation)"
#    이 함수는 {코드: "약어 (풀네임)"} 딕셔너리를 만들어 준다.
# ──────────────────────────────────────────────────────────────────────
def load_condition_map(csv_path):
    code_to_name = {}
    if not os.path.exists(csv_path):
        return code_to_name
    with open(csv_path, encoding="utf-8-sig") as f:
        next(f)  # 첫 줄(헤더: Acronym,Name,Snomed_CT) 건너뛰기
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                acronym, full_name, code = parts[0], parts[1], parts[2]
                code_to_name[code] = f"{acronym} ({full_name})"
    return code_to_name


# ──────────────────────────────────────────────────────────────────────
# 3. 레코드 이름 -> 실제 파일 경로 찾기
#    사용자가 "JS00007"처럼 이름만 줘도, 45000여 폴더 중 어디 있는지 찾아준다.
#    wfdb는 확장자 없는 경로를 받으므로 .hea/.mat 를 뗀 경로를 돌려준다.
# ──────────────────────────────────────────────────────────────────────
def resolve_record_path(user_input):
    # (a) 사용자가 "01/010/JS00002" 처럼 경로를 직접 준 경우
    direct = os.path.join(WFDB_DIR, user_input)
    if os.path.exists(direct + ".hea"):
        return direct

    # (b) "JS00007" 처럼 이름만 준 경우 -> 하위 폴더 전체에서 검색
    hits = glob.glob(os.path.join(WFDB_DIR, "**", user_input + ".hea"),
                     recursive=True)
    if hits:
        return hits[0][:-4]  # ".hea" 4글자 제거

    raise FileNotFoundError(f"레코드를 찾을 수 없습니다: {user_input}")


# ──────────────────────────────────────────────────────────────────────
# 4. .hea 주석에서 환자 메타데이터(나이/성별/진단) 뽑아 보기 좋게 만들기
# ──────────────────────────────────────────────────────────────────────
def format_metadata(record, code_map):
    info = {}
    for line in record.comments:          # 예: "Age: 85", "Dx: 164889003,59118001"
        if ":" in line:
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()

    age = info.get("Age", "?")
    sex = info.get("Sex", "?")

    # 진단 코드들을 사람이 읽을 수 있는 이름으로 변환
    dx_codes = info.get("Dx", "").split(",")
    dx_names = [code_map.get(c.strip(), c.strip()) for c in dx_codes if c.strip()]
    dx_text = ", ".join(dx_names) if dx_names else "None"

    return age, sex, dx_text


# ──────────────────────────────────────────────────────────────────────
# 5. 핵심: 12-lead ECG를 4x3 격자로 그리기
# ──────────────────────────────────────────────────────────────────────
def plot_ecg(record_path, code_map):
    # (1) WFDB 레코드 읽기 — .hea와 .mat을 자동으로 짝지어 읽는다
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal      # shape = (5000, 12) : 5000샘플 x 12리드, 단위 mV
    fs = record.fs                # 샘플링 레이트 = 500 Hz
    lead_names = record.sig_name  # ['I','II','III','aVR',...,'V6']
    n_samples = signal.shape[0]

    # (2) x축(시간) 만들기: 샘플 번호 / 초당 샘플수 = 초 단위 시간
    #     0번 샘플=0초, 500번 샘플=1초 ... 5000번 샘플=10초
    time = np.arange(n_samples) / fs

    # (3) 환자 정보 추출
    age, sex, dx_text = format_metadata(record, code_map)
    plt.hist(age, bins = 15, color = 'skyblue', edgecolor = 'black')
    plt.hist(sex, bins = 15, color = 'skyblue', edgecolor = 'black')

    # (4) 4행 x 3열 격자 = 12칸. 각 칸에 리드 하나씩.
    #     sharex=True : 모든 그래프가 같은 시간축을 공유
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()  # 4x3 2차원 배열을 길이 12 1차원으로 펴기

    for i, ax in enumerate(axes):
        ax.plot(time, signal[:, i], linewidth=0.8, color="black")
        ax.set_title(lead_names[i], fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, which="both", color="red", alpha=0.3, linewidth=0.5)
        ax.set_ylabel("mV", fontsize=8)

    # 맨 아랫줄에만 시간축 라벨
    for ax in axes[-3:]:
        ax.set_xlabel("Time (s)", fontsize=9)

    # (5) 전체 제목에 환자 정보 표시
    rec_name = os.path.basename(record_path)
    fig.suptitle(
        f"12-Lead ECG  |  {rec_name}  |  Age {age}, {sex}\nDx: {dx_text}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 공간 확보

    # (6) 파일로 저장 + 화면에 표시
    out_path = os.path.join(os.path.dirname(__file__), f"{rec_name}_ecg.png")
    fig.savefig(out_path, dpi=120)
    print(f"[저장됨] {out_path}")
    print(f"  - 신호 크기: {signal.shape}  (샘플 x 리드)")
    print(f"  - 측정 시간: {n_samples / fs:.1f}초 @ {fs}Hz")
    print(f"  - 진단: {dx_text}")

    plt.show()  # 창이 뜨는 환경이면 그래프 표시


# ──────────────────────────────────────────────────────────────────────
# 6. 프로그램 시작점
# ──────────────────────────────────────────────────────────────────────
def main():
    # 명령줄 인자가 있으면 그걸, 없으면 기본 샘플 사용
    user_input = sys.argv[1] if len(sys.argv) > 1 else "JS00001"

    code_map = load_condition_map(COND_CSV)
    record_path = resolve_record_path(user_input)
    plot_ecg(record_path, code_map)
    
if __name__ == "__main__":
    main()    