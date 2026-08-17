"""
삼차 스플라인 보간 (Cubic Spline Interpolation, numpy 전용)

구간별 3차 다항식으로 점을 잇는다. 이차 스플라인과 달리
곡률(2차 도함수)까지 연속(C2)이라 훨씬 매끄럽고 오버슈트가 적다.

경계 조건(bc):
    'natural'  : 양 끝의 2차 도함수를 0 으로 (열린 곡선)
    'periodic' : 시작점과 끝점이 같은 닫힌 곡선. 이음매에서 기울기·곡률 연속.

매듭에서의 2차 도함수 M_i 를 선형계로 풀어 계수를 구성한다.
점 개수가 수십 개 수준이라 밀집행렬 + np.linalg.solve 로 충분하다.
"""

import numpy as np


class CubicSpline:
    def __init__(self, t, y, bc="natural"):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if t.size != y.size or t.size < 3:
            raise ValueError("점이 최소 3개 필요하고 t, y 길이가 같아야 합니다.")

        self.bc = bc
        self.t = t
        self.y = y
        self.h = np.diff(t)

        if bc == "periodic":
            self.M = self._solve_periodic()
        else:
            self.M = self._solve_natural()

    def _solve_natural(self):
        n = self.t.size
        A = np.zeros((n, n))
        d = np.zeros(n)
        A[0, 0] = 1.0                    # M_0 = 0
        A[-1, -1] = 1.0                  # M_{n-1} = 0
        for i in range(1, n - 1):
            A[i, i - 1] = self.h[i - 1]
            A[i, i] = 2.0 * (self.h[i - 1] + self.h[i])
            A[i, i + 1] = self.h[i]
            d[i] = 6.0 * ((self.y[i + 1] - self.y[i]) / self.h[i]
                          - (self.y[i] - self.y[i - 1]) / self.h[i - 1])
        return np.linalg.solve(A, d)

    def _solve_periodic(self):
        # 닫힌 곡선: 마지막 점 == 첫 점 이라고 가정. 독립 매듭은 0..n-2.
        m = self.t.size - 1              # 독립 매듭 개수
        h = self.h                       # 길이 m
        A = np.zeros((m, m))
        d = np.zeros(m)
        for i in range(m):
            hm = h[i - 1]                # h_{i-1} (i=0 이면 마지막 구간, wrap)
            hi = h[i]
            A[i, (i - 1) % m] += hm
            A[i, i] += 2.0 * (hm + hi)
            A[i, (i + 1) % m] += hi
            ynext = self.y[i + 1]
            yprev = self.y[i - 1] if i > 0 else self.y[m - 1]
            d[i] = 6.0 * ((ynext - self.y[i]) / hi
                          - (self.y[i] - yprev) / hm)
        Msub = np.linalg.solve(A, d)
        return np.concatenate([Msub, Msub[:1]])   # M_{n-1} = M_0

    def __call__(self, tq):
        tq = np.asarray(tq, dtype=float)
        scalar = tq.ndim == 0
        tq = np.atleast_1d(tq)

        i = np.searchsorted(self.t, tq, side="right") - 1
        i = np.clip(i, 0, self.t.size - 2)

        h = self.h[i]
        a = self.t[i + 1] - tq
        b = tq - self.t[i]
        out = (self.M[i] * a ** 3 / (6.0 * h)
               + self.M[i + 1] * b ** 3 / (6.0 * h)
               + (self.y[i] / h - self.M[i] * h / 6.0) * a
               + (self.y[i + 1] / h - self.M[i + 1] * h / 6.0) * b)
        return float(out[0]) if scalar else out
