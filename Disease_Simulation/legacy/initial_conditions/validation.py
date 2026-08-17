"""
validation.py
================================================================
초기 상태의 물리적 타당성 검증 + 요약 통계.

보장해야 할 불변식:
  S >= 0,  I >= 0,  R >= 0        (음수 인구는 없다)
  N = S + I + R                    (인구 보존)
"""

import numpy as np


def validate_and_conserve(S, I, R, atol=1e-6):
    """비음수 보정 후 N=S+I+R 을 재계산해 인구 보존을 보장한다.

    수치적으로 아주 작은 음수(부동소수 오차)는 0 으로 클립한다. 프리셋들이
    이미 S 재고 한도로 잘라 두므로 정상 경로에서는 클립이 거의 일어나지 않는다."""
    np.clip(S, 0.0, None, out=S)
    np.clip(I, 0.0, None, out=I)
    np.clip(R, 0.0, None, out=R)

    N = S + I + R

    # 불변식 검증(문제가 있으면 조용히 넘어가지 않고 즉시 알림)
    assert np.all(S >= 0), "S 에 음수가 있음"
    assert np.all(I >= 0), "I 에 음수가 있음"
    assert np.all(R >= 0), "R 에 음수가 있음"
    assert np.allclose(N, S + I + R, atol=atol), "N != S+I+R (인구 미보존)"
    return S, I, R, N


def summarize(S, I, R, N):
    """초기 상태 요약 통계를 dict 로 반환."""
    tot = float(N.sum())
    tS, tI, tR = float(S.sum()), float(I.sum()), float(R.sum())
    return {
        "total_S": tS,
        "total_I": tI,
        "total_R": tR,
        "total_population": tot,
        "infected_pct": (tI / tot * 100.0) if tot > 0 else 0.0,
        "recovered_pct": (tR / tot * 100.0) if tot > 0 else 0.0,
        "susceptible_pct": (tS / tot * 100.0) if tot > 0 else 0.0,
        "peak_density": float(N.max()),
        "num_infected_cells": int((I > 0).sum()),
    }


def print_summary(stats, title="초기 상태 요약"):
    """요약 통계를 보기 좋게 출력."""
    print(f"── {title} " + "─" * max(0, 40 - len(title)))
    print(f"  전체 인구      : {stats['total_population']:>14,.0f} 명")
    print(f"  감염가능 S     : {stats['total_S']:>14,.0f} 명 "
          f"({stats['susceptible_pct']:5.2f}%)")
    print(f"  감염중   I     : {stats['total_I']:>14,.0f} 명 "
          f"({stats['infected_pct']:5.2f}%)")
    print(f"  회복/면역 R    : {stats['total_R']:>14,.0f} 명 "
          f"({stats['recovered_pct']:5.2f}%)")
    print(f"  감염 시작 칸 수 : {stats['num_infected_cells']:>14,d} 칸")
    print(f"  최대 인구밀도  : {stats['peak_density']:>14,.0f} 명/km²")
