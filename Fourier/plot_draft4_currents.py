"""
Draft4.txt (시간영역 과도응답: time | I(C1) | I(C2) | I(L2)) — TMD 회로의 세 전류.

표기 (우리 연립 미분방정식 변수와 일치):
    I(C1) = i1          (메인 루프 전류)
    I(C2) = i1 - i2     (공유가지 = C2 통과 전류)
    I(L2) = i2          (흡진기 인덕터 전류)
세 곡선을 하나의 플롯에 그린다.
"""

import numpy as np
import matplotlib.pyplot as plt

t, iC1, iC2, iL2 = [], [], [], []
with open("Draft4.txt", encoding="latin-1") as f:
    next(f)                                   # 헤더 건너뛰기
    for line in f:
        p = line.split()
        if len(p) >= 4:
            t.append(float(p[0]))
            iC1.append(float(p[1]))
            iC2.append(float(p[2]))
            iL2.append(float(p[3]))

t = np.array(t)
iC1, iC2, iL2 = np.array(iC1), np.array(iC2), np.array(iL2)

print(f"데이터 점 {len(t)}개,  시간 0 ~ {t.max():.3g} s")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t * 1e3, iC1, "-", lw=1.4, color="crimson",   label="$i_1$  = I(C1)")
ax.plot(t * 1e3, iC2, "-", lw=1.4, color="seagreen",  label="$i_1 - i_2$  = I(C2)")
ax.plot(t * 1e3, iL2, "-", lw=1.4, color="royalblue", label="$i_2$  = I(L2)")
ax.axhline(0, color="0.7", lw=0.8)

ax.set_xlabel("time [ms]")
ax.set_ylabel("current [A]")
ax.set_title("TMD circuit transient: three branch currents")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()
fig.savefig("draft4_currents.png", dpi=130)
print("그래프를 draft4_currents.png 로 저장했습니다.")
