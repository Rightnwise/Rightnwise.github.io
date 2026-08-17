"""
Draft2.txt (회로 주파수 응답: Freq | (magnitude dB, phase deg)) 에서
위상은 무시하고 '진폭'만 주파수에 대해 그린다.

단위 변환: dB -> 암페어
    mag[dB] = 20 * log10(I / 1A)   =>   I = 10^(mag_dB / 20)
공진(resonance)은 진폭이 최대가 되는 주파수에서 피크로 나타난다.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

PATH = "Draft2.txt"

freqs, amps_db = [], []
num = r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?"     # 부호/소수/지수 포함 실수
pat = re.compile(rf"({num})\s*\(\s*({num})\s*dB", re.IGNORECASE)

with open(PATH, encoding="latin-1") as f:
    for line in f:
        m = pat.search(line)
        if m:
            freqs.append(float(m.group(1)))
            amps_db.append(float(m.group(2)))

freqs = np.array(freqs)
amps_db = np.array(amps_db)
amps_A = 10 ** (amps_db / 20.0)          # dB -> 암페어

# 공진점 (진폭 최대)
i_peak = int(np.argmax(amps_A))
f_res, i_res = freqs[i_peak], amps_A[i_peak]

print(f"데이터 점 {len(freqs)}개,  주파수 {freqs.min():.3g} ~ {freqs.max():.3g} Hz")
print(f"공진 주파수 ~ {f_res:.6g} Hz,  최대 진폭 = {i_res:.4g} A")

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(freqs, amps_A, "-", lw=1.8, color="crimson", label="|I(L1)|")
ax.plot(f_res, i_res, "o", color="navy", ms=9, zorder=5,
        label=f"resonance ~ {f_res:.4g} Hz")
ax.axvline(f_res, color="navy", ls="--", lw=1, alpha=0.6)

ax.set_xscale("log")                     # 주파수 범위가 넓으면 로그축이 보기 좋음
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude  |I(L1)|  [A]")
ax.set_title("Frequency response: current amplitude (resonance)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()
fig.savefig("amplitude_response.png", dpi=130)
print("그래프를 amplitude_response.png 로 저장했습니다.")
plt.show()
