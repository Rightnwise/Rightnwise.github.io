"""
Draft3 (time | I(L1)) — 임펄스(펄스) 입력에 대한 인덕터 전류의 과도 응답.

직렬 RLC:  R1 = 100 ohm,  C1 = 0.1 uF,  L1 = 10 mH
입력: PULSE(0 5 10u 1n 1n 2u 10m)  (5V, t=10us 에 폭 2us 펄스 -> 거의 임펄스)

이론 공진 주파수  f0 = 1/(2*pi*sqrt(L*C)) ~ 5.03 kHz
임펄스가 들어오면 회로는 f0 부근에서 '울림(ringing)' 하며 지수적으로 감쇠한다.
"""

import numpy as np
import matplotlib.pyplot as plt

PATH = "Draft3"
R, L, C = 100.0, 10e-3, 0.1e-6
f0 = 1 / (2 * np.pi * np.sqrt(L * C))

t, i = [], []
with open(PATH, encoding="latin-1") as f:
    next(f)                                   # 헤더 (time  I(L1))
    for line in f:
        parts = line.split()
        if len(parts) == 2:
            try:
                t.append(float(parts[0])); i.append(float(parts[1]))
            except ValueError:
                pass

t = np.array(t) * 1e3                          # s -> ms
i = np.array(i) * 1e3                          # A -> mA

print(f"점 {len(t)}개,  t = {t.min():.3g} ~ {t.max():.3g} ms")
print(f"이론 공진 f0 = {f0:.4g} Hz,  |I| 최대 = {np.abs(i).max():.4g} mA")

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t, i, "-", lw=1.4, color="crimson", label="I(L1)")
ax.axhline(0, color="0.5", lw=0.8)

ax.set_xlabel("Time [ms]")
ax.set_ylabel("Inductor current  I(L1)  [mA]")
ax.set_title("Series RLC impulse response: damped ringing of I(L1)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# PPT 용 파라미터 박스
txt = (f"R = 100 $\\Omega$\nL = 10 mH\nC = 0.1 $\\mu$F\n"
       f"impulse @ 10 $\\mu$s\n$f_0=1/2\\pi\\sqrt{{LC}}\\approx$ {f0/1e3:.2f} kHz")
ax.text(0.985, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", fc="#fff7e6", ec="orange", alpha=0.95))

fig.tight_layout()
fig.savefig("impulse_response.png", dpi=130)
print("그래프를 impulse_response.png 로 저장했습니다.")
plt.show()
