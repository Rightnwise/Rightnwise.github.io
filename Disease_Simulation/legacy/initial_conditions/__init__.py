"""
initial_conditions
================================================================
2D 반응-확산 SIR 시뮬레이션을 위한 '모듈식 초기 조건 시스템'.

이 패키지의 목표는 단순히 배열을 채우는 것이 아니라, 서로 다른 출발
조건이 감염 확산에 어떤 영향을 주는지 실험할 수 있는 도구를 제공하는 것.

핵심 API:
    S, I, R, N = create_initial_conditions(config)

config 딕셔너리 하나로 아래를 조합한다:
    · 인구밀도 지도       (population_maps.py)   : population_type
    · 초기 발생 프리셋     (presets.py)           : initial_condition_type
    · 사전 면역/백신      (presets.apply_immunity): recovered_fraction 등
    · 검증 & 요약통계     (validation.py)
    · 시각화             (visualization.py)

확장성(설계 의도):
    반환하는 (S, I, R, N) 은 그대로 SpatialSIR 모델에 주입할 수 있고,
    아래 확장과도 자연스럽게 맞물린다.
      - 공간적 β 지도        : population_maps 와 같은 방식으로 β(x,y) 생성
      - 공간적 확산계수      : mobility/확산 마스크로 D(x,y)
      - 격리구역             : vaccinated_regions 와 동일 패턴의 마스크
      - 백신 시뮬레이션      : recovered_fraction / vaccinated_regions
      - SEIR 확장            : create 에서 E 칸막이만 추가하면 됨(구조 동일)
"""

import numpy as np

from .population_maps import build_population, POPULATION_MAPS
from .presets import OUTBREAK_PRESETS, apply_immunity
from .validation import validate_and_conserve, summarize, print_summary
from .visualization import plot_initial_conditions

__all__ = [
    "create_initial_conditions",
    "summarize",
    "print_summary",
    "plot_initial_conditions",
    "DEFAULT_CONFIG",
]

# 바꾸기 쉬운 기본 설정: config 로 넘긴 값이 이 위에 덮인다
DEFAULT_CONFIG = {
    "grid_size": (100, 100),              # (ny, nx) 행,열
    "population_type": "uniform",         # 인구밀도 지도 종류
    "base_density": 500.0,                # 기준 인구밀도 [명/km²]
    "initial_condition_type": "center_outbreak",
    "initial_infected_fraction": 0.01,    # 씨앗 지역에서 감염되는 인구 비율
    "outbreak_radius": 5,                 # 씨앗 반경 [칸]
    "recovered_fraction": 0.0,            # 전역 사전 면역 비율
    "random_seed": 42,                    # 재현성
}


def create_initial_conditions(config=None):
    """config 로부터 초기 상태 (S, I, R, N) 를 만든다.

    처리 순서(각 단계는 인구를 보존):
      1) 인구밀도 지도 N₀ 생성            (population_maps)
      2) S=N₀, I=0, R=0 로 시작
      3) 사전 면역 적용  S→R              (presets.apply_immunity)
      4) 초기 발생 적용  S→I              (presets.OUTBREAK_PRESETS)
      5) 비음수·보존 검증 후 N=S+I+R 확정 (validation)

    Returns:
        S, I, R, N : 각각 (ny, nx) 2D numpy 배열
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    # 1) 인구밀도 지도
    N0 = build_population(cfg)

    # 2) 전원 감염가능에서 출발
    S = N0.copy()
    I = np.zeros_like(N0)
    R = np.zeros_like(N0)

    # 3) 사전 면역/백신 (S→R)
    S, R = apply_immunity(S, R, N0, cfg)

    # 4) 초기 발생 (S→I)
    ic_type = cfg.get("initial_condition_type", "center_outbreak")
    if ic_type not in OUTBREAK_PRESETS:
        raise ValueError(f"알 수 없는 initial_condition_type: {ic_type!r} "
                         f"(가능: {list(OUTBREAK_PRESETS)})")
    S, I, R = OUTBREAK_PRESETS[ic_type](S, I, R, N0, cfg)

    # 5) 검증 + 인구 보존 확정
    S, I, R, N = validate_and_conserve(S, I, R)
    return S, I, R, N
