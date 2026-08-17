"""
Draft1.txt (with TMD, I(C1)) vs Draft2.txt (without TMD, I(L1))
주파수 응답: Freq | (magnitude dB, phase deg). 위상 무시, dB -> 암페어.
    I = 10^(mag_dB / 20)

출력:
  amplitude_response_draft1.png  - Draft1 단독 (축 범위를 Draft2 와 동일하게 맞춤)
  amplitude_overlay.png          - Draft1 + Draft2 를 한 그래프에 겹침
"""

import re
import numpy as np
import matplotlib.pyplot as plt

num = r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?"
pat = re.compile(rf"({num})\s*\(\s*({num})\s*dB", re.IGNORECASE)


def load(path):
    """주파수 응답 파일 -> (freq[Hz], amplitude[A])"""
    freqs, amps_db = [], []
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = pat.search(line)
            if m:
                freqs.append(float(m.group(1)))
                amps_db.append(float(m.group(2)))
    freqs = np.array(freqs)
    amps_A = 10 ** (np.array(amps_db) / 20.0)
    return freqs, amps_A


f1, a1 = load("Draft1.txt")    # TMD 있음  (I(C1))
f2, a2 = load("Draft2.txt")    # TMD 없음  (I(L1))

p1 = int(np.argmax(a1)); p2 = int(np.argmax(a2))
print(f"Draft1 (TMD 있음): 피크 {a1[p1]:.4g} A @ {f1[p1]:.4g} Hz")
print(f"Draft2 (TMD 없음): 피크 {a2[p2]:.4g} A @ {f2[p2]:.4g} Hz")

# amplitude_response.png(원본) 과 '똑같은' 축 범위 추출:
# 동일한 figsize/log-scale 로 Draft2 를 autoscale 시킨 뒤 그 한계를 그대로 가져온다.
_rf, _rax = plt.subplots(figsize=(11, 6))
_rax.plot(f2, a2, "-", lw=1.8)
_rax.plot(f2[p2], a2[p2], "o", ms=9)
_rax.set_xscale("log")
_rf.canvas.draw()
XLIM = _rax.get_xlim()
YLIM = _rax.get_ylim()
plt.close(_rf)

# ── 그림 1: Draft1 단독, 축 범위/스케일을 원본과 동일하게 (그래프만 교체용) ──
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(f1, a1, "-", lw=1.8, color="crimson", label="|I(C1)|")
ax.set_xscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude  |I(C1)|  [A]")
ax.set_title("Frequency response: current amplitude (resonance)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("amplitude_response_draft1.png", dpi=130)
print("amplitude_response_draft1.png 저장 (축 범위 원본과 동일)")

# ── 그림 2: 두 곡선 겹쳐 그리기 ──
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(f2, a2, "-", lw=2, color="royalblue",
        label="without TMD  |I(L1)|  (single peak)")
ax.plot(f1, a1, "-", lw=2, color="crimson",
        label="with TMD  |I(C1)|  (notch + two peaks)")
ax.set_xscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude  [A]")
ax.set_title("Frequency response: current amplitude (resonance)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("amplitude_overlay.png", dpi=130)
print("amplitude_overlay.png 저장 (Draft1 + Draft2 겹침)")
