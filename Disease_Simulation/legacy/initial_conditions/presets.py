"""
presets.py
================================================================
초기 발생(outbreak) 및 초기 면역(immunity) 배치 프리셋.

핵심 원칙 — 인구 보존:
  초기화는 '새 사람을 만드는' 것이 아니라, 이미 존재하는 인구 N 을
  칸별로 S/I/R 세 칸막이로 나누는 일이다. 따라서 감염자를 심을 때는
  반드시 그 지역의 S 에서 빼서 I 로 옮기고(S→I), 사전면역을 줄 때는
  S 에서 빼서 R 로 옮긴다(S→R). 그러면 N = S+I+R 이 항상 보존된다.

좌표 규약: seed 좌표는 (x, y) = (열, 행) 픽셀 인덱스.
"""

import numpy as np


# ---------------- 공통 도구 ----------------
def _disc_mask(shape, cx, cy, radius):
    """(cx, cy) 중심 반경 radius 안의 원형 마스크(불리언)."""
    ny, nx = shape
    yy, xx = np.ogrid[0:ny, 0:nx]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def seed_region(S, I, N, cx, cy, radius, fraction):
    """원형 영역 안에서 '지역 인구의 fraction' 만큼 S→I 로 감염 심기.

    fraction=0.01 이면 그 지역 사람의 1% 가 초기 감염자가 된다는 뜻.
    S 재고를 넘지 않도록 잘라서(S 보다 많이 감염 못 시킴) 음수를 막는다."""
    mask = _disc_mask(S.shape, cx, cy, radius)
    amount = np.where(mask, fraction * N, 0.0)
    amount = np.minimum(amount, S)          # 인구 보존 & 비음수 보장
    S -= amount
    I += amount
    return S, I


# ---------------- (A) 중심 발생 ----------------
def center_outbreak(S, I, R, N, config):
    """A. 격자 중앙 한 곳에서 소규모 감염 시작.

    역학적 의미: '환자 0번'이 도심 한복판에서 발생한 전형적 시나리오."""
    ny, nx = S.shape
    r = config.get("outbreak_radius", 5)
    frac = config.get("initial_infected_fraction", 0.01)
    S, I = seed_region(S, I, N, (nx - 1) / 2.0, (ny - 1) / 2.0, r, frac)
    return S, I, R


# ---------------- (B) 다중 발생 ----------------
def multiple_outbreaks(S, I, R, N, config):
    """B. 사용자가 지정한 여러 좌표에서 동시에 감염 시작.

    역학적 의미: 여러 도시에 동시 유입(예: 공항 여러 곳)된 상황. 각 씨앗은
    반경·감염비율을 개별 지정할 수 있다.
    config["outbreak_seeds"] = [{"x","y","radius","fraction"}, ...]"""
    ny, nx = S.shape
    seeds = config.get("outbreak_seeds")
    if not seeds:
        # 기본: 네 귀퉁이 근처 + 중앙
        seeds = [{"x": nx * 0.25, "y": ny * 0.25},
                 {"x": nx * 0.75, "y": ny * 0.25},
                 {"x": nx * 0.25, "y": ny * 0.75},
                 {"x": nx * 0.75, "y": ny * 0.75}]
    r_def = config.get("outbreak_radius", 4)
    f_def = config.get("initial_infected_fraction", 0.01)
    for s in seeds:
        S, I = seed_region(S, I, N, s["x"], s["y"],
                           s.get("radius", r_def), s.get("fraction", f_def))
    return S, I, R


# ---------------- (C) 무작위 발생 ----------------
def random_outbreaks(S, I, R, N, config):
    """C. 무작위 위치 여러 곳에서 감염 시작(재현 가능).

    역학적 의미: 어디서 터질지 모르는 확률적 유입. random_seed 를 고정하면
    같은 배치가 재현되어 실험 비교가 가능하다."""
    ny, nx = S.shape
    rng = np.random.default_rng(config.get("random_seed"))
    num = int(config.get("num_random_seeds", 5))
    r = config.get("outbreak_radius", 3)
    frac = config.get("initial_infected_fraction", 0.01)
    xs = rng.integers(0, nx, size=num)
    ys = rng.integers(0, ny, size=num)
    for cx, cy in zip(xs, ys):
        S, I = seed_region(S, I, N, cx, cy, r, frac)
    return S, I, R


# ---------------- (E) 사전 면역 / 백신 접종 ----------------
def apply_immunity(S, R, N, config):
    """E. 일부 인구를 시작부터 R(면역)로 둔다 → S→R.

    역학적 의미: 사전 면역/백신 접종. 감염가능자(S)를 줄여 실효 재생산수를
    낮춘다(집단면역 실험). 두 가지 방식을 지원:
      · recovered_fraction : 전 지역 균일하게 그 비율만큼 면역
      · vaccinated_regions : 특정 영역만 면역(지역 접종 캠페인)
        [{"x","y","radius","fraction"}, ...]"""
    frac = config.get("recovered_fraction", 0.0)
    if frac > 0:
        amount = np.minimum(frac * N, S)
        S -= amount
        R += amount

    for reg in config.get("vaccinated_regions", []):
        mask = _disc_mask(S.shape, reg["x"], reg["y"], reg.get("radius", 5))
        amount = np.where(mask, reg.get("fraction", 0.5) * N, 0.0)
        amount = np.minimum(amount, S)
        S -= amount
        R += amount
    return S, R


# 문자열 → 발생 프리셋 등록부
OUTBREAK_PRESETS = {
    "center_outbreak": center_outbreak,
    "multiple_outbreaks": multiple_outbreaks,
    "random_outbreaks": random_outbreaks,
}
