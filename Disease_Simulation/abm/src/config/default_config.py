"""
시뮬레이션 설정 (SimConfig).

일(day) 단위로 파라미터를 받고, 내부에서 틱(tick=낮/밤) 단위로 변환한다.
기본값은 legacy/nyc_metapop_sird_final.py 의 baseline 과 동일하다.
(개입 파라미터는 모두 '없음' 상태가 기본 → baseline 재현)
"""
from dataclasses import dataclass, fields, asdict
from typing import Optional


@dataclass
class SimConfig:
    scenario_name: str = "baseline"

    # --- 역학 파라미터(일 단위) ---
    beta: float = 0.10                 # 감염 계수
    latent_days: int = 4               # 잠복기(무증상 전파) 일수
    recovery_days: int = 14            # 감염→회복/사망 판정까지 일수
    immunity_days: Optional[int] = 60  # 회복 후 면역 지속 일수(이후 다시 S). None=영구면역
                                       #   60일 이하 → 면역이 유행 중 풀려 재유행(endemic) 발생
    fatality_rate: float = 0.02        # 치사율
    initial_infected: int = 5          # 초기 감염자 수(소수, 한 지점에서 발생)
    seed_location: str = "north"       # 최초 발생 지점: "north"/"south"/"center"/노드idx(int)
    simulation_days: int = 300         # 최대 시뮬레이션 일수(유행 종료 시 조기 중단)

    # --- 이동/개입 ---
    commute_probability: float = 1.0   # 낮에 출근할 기본 확률(1.0 = 전원 출근)
    commute_reduction: float = 0.0     # 봉쇄 시 이동 감소율(lockdown_day 부터 적용)
    lockdown_day: Optional[int] = None  # 봉쇄 시작일(None=봉쇄 없음)
    vaccination_rate: float = 0.0      # 시작 전 S→R 로 접종할 인구 비율
    vaccination_strategy: str = "random"   # "random" | "high_density_first"
    isolation_effectiveness: float = 1.0   # 증상자 자가격리 성공률(1.0=항상 격리)

    # --- 재현성/구조 ---
    random_seed: int = 42
    pop_per_agent: int = 1000          # 에이전트 1명 = 실제 N명
    distance_decay_alpha: float = 4.0  # 출퇴근 중력모형 거리 감쇠 지수(4=국소적→동심원 확산)
    ticks_per_day: int = 2             # 낮/밤

    # ---- 틱 환산(내부용) ----
    @property
    def latent_ticks(self):
        return self.latent_days * self.ticks_per_day

    @property
    def recovery_ticks(self):
        return self.recovery_days * self.ticks_per_day

    @property
    def immunity_ticks(self):
        if self.immunity_days is None or self.immunity_days <= 0:
            return None                       # 영구 면역
        return self.immunity_days * self.ticks_per_day

    @property
    def max_ticks(self):
        return self.simulation_days * self.ticks_per_day

    def to_dict(self):
        return asdict(self)


def make_config(config=None, **overrides):
    """dict / SimConfig / 부분 dict 를 받아 완성된 SimConfig 로 반환."""
    if isinstance(config, SimConfig):
        base = config.to_dict()
    elif isinstance(config, dict):
        base = dict(config)
    else:
        base = {}
    base.update(overrides)
    valid = {f.name for f in fields(SimConfig)}
    unknown = set(base) - valid
    if unknown:
        raise ValueError(f"알 수 없는 config 키: {unknown}")
    return SimConfig(**base)


# baseline 기본값 dict(참고/탐색용)
DEFAULT_CONFIG = SimConfig().to_dict()
