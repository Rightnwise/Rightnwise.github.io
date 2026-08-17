"""
population_maps.py
================================================================
인구밀도 지도 N(x, y) 를 만드는 함수 모음.

역학적 의미:
  N 은 각 칸(1km²)에 사는 사람 수(밀도)다. 감염 동역학 β·S·I/N 에서
  분모로 쓰이므로, 인구가 밀집한 곳일수록 같은 감염자 수라도 접촉이
  잦아져 국소 전파가 빨라진다. 도시(고밀도)–농촌(저밀도) 구조를 지도로
  주면 "어디서 터지느냐"가 유행 규모에 어떻게 영향을 주는지 실험할 수 있다.

모든 함수는 (ny, nx) 모양의 2D 인구밀도 배열을 반환한다.
스칼라 파라미터만 바꿔 쉽게 다른 지형을 만들 수 있도록 설계했다.
"""

import numpy as np


def _box_blur(f):
    """무유출 경계로 3x3 평균 한 번(난수 밀도장을 매끄럽게 만들 때 사용)."""
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2]
            + fp[1:-1, 2:] + f) / 5.0


def uniform(shape, base_density=500.0, **kw):
    """균일 인구밀도.

    역학적 의미: 지형 효과가 전혀 없는 '실험실' 조건. 초기 발생 위치·확산
    계수 등 다른 변수의 순수 효과를 보고 싶을 때의 기준선(baseline)."""
    return np.full(shape, float(base_density), dtype=float)


def high_density_center(shape, base_density=100.0, peak_density=1200.0,
                        sigma=None, **kw):
    """중심 고밀도(단일 대도시) — 가우시안 봉우리 하나.

    역학적 의미: 하나의 거대 도시. 중심에서 병이 터지면 폭발적으로 커지고,
    외곽에서 터지면 저밀도 지역을 지나 도심에 도달할 때까지 시간이 걸린다."""
    ny, nx = shape
    if sigma is None:
        sigma = min(nx, ny) / 6.0
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    bump = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    return base_density + (peak_density - base_density) * bump


def multiple_centers(shape, base_density=80.0, peak_density=1200.0,
                     centers=None, sigma=None, **kw):
    """여러 인구 중심(다핵 도시권) — 가우시안 봉우리 여러 개의 합.

    역학적 의미: 수도권처럼 여러 도시가 떨어져 있는 구조. 도시 간
    '저밀도 회랑'이 자연적인 확산 장벽/지연 역할을 한다.
    centers: [(x, y), ...] 픽셀 좌표 목록."""
    ny, nx = shape
    if sigma is None:
        sigma = min(nx, ny) / 10.0
    if centers is None:
        centers = [(nx * 0.25, ny * 0.30),
                   (nx * 0.72, ny * 0.35),
                   (nx * 0.50, ny * 0.75)]
    yy, xx = np.mgrid[0:ny, 0:nx]
    field = np.full(shape, float(base_density), dtype=float)
    for cx, cy in centers:
        field += (peak_density - base_density) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    return field


def random_field(shape, base_density=100.0, peak_density=1200.0,
                 smoothness=8, random_seed=None, **kw):
    """무작위(하지만 매끄러운) 인구밀도장.

    역학적 의미: 실제 지형처럼 불규칙하게 분포한 마을/도시. 난수를 여러 번
    평활화해 자연스러운 얼룩무늬 밀도를 만든다. random_seed 로 재현 가능.
    smoothness 가 클수록 큰 덩어리(부드러움)."""
    rng = np.random.default_rng(random_seed)
    noise = rng.random(shape)
    for _ in range(int(smoothness)):
        noise = _box_blur(noise)
    span = np.ptp(noise)
    noise = (noise - noise.min()) / (span + 1e-12)  # 0~1 정규화
    return base_density + (peak_density - base_density) * noise


# 문자열 → 함수 등록부(레지스트리): config 로 손쉽게 고르게 한다
POPULATION_MAPS = {
    "uniform": uniform,
    "high_density_center": high_density_center,
    "multiple_centers": multiple_centers,
    "random_field": random_field,
}


def build_population(config):
    """config 에 맞는 인구밀도 지도 N 을 만든다."""
    shape = tuple(config["grid_size"])
    ptype = config.get("population_type", "uniform")
    if ptype not in POPULATION_MAPS:
        raise ValueError(f"알 수 없는 population_type: {ptype!r} "
                         f"(가능: {list(POPULATION_MAPS)})")
    params = dict(config.get("population_params", {}))
    # 편의: 최상위 config 의 공통 키를 넘겨준다(함수는 **kw 로 무시 가능)
    params.setdefault("base_density", config.get("base_density", 500.0))
    params.setdefault("random_seed", config.get("random_seed"))
    return POPULATION_MAPS[ptype](shape, **params).astype(float)
