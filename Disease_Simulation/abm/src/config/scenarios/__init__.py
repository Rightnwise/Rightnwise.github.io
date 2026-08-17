"""
시나리오 정의 = baseline SimConfig 위에 덮어쓸 override dict 모음.

commute_reduction 은 lockdown_day 부터 적용되므로,
'상시 이동 감소' 시나리오는 lockdown_day=0 으로 처음부터 적용한다.
"""

SCENARIOS = {
    "baseline": {},
    "low_beta": {"beta": 0.05},
    "high_beta": {"beta": 0.15},
    # 상시 이동 감소(처음부터)
    "commute_50_reduction": {"lockdown_day": 0, "commute_reduction": 0.5},
    "commute_80_reduction": {"lockdown_day": 0, "commute_reduction": 0.8},
    # 시점별 봉쇄(이동 70% 감소)
    "early_lockdown": {"lockdown_day": 20, "commute_reduction": 0.7},
    "late_lockdown": {"lockdown_day": 60, "commute_reduction": 0.7},
    # 백신(초기 인구 50% 면역)
    "vaccine_random_50": {"vaccination_rate": 0.5, "vaccination_strategy": "random"},
    "vaccine_high_density_50": {"vaccination_rate": 0.5,
                                "vaccination_strategy": "high_density_first"},
}


def get_scenario_config(name):
    """시나리오 이름 → override dict(scenario_name 포함)."""
    if name not in SCENARIOS:
        raise KeyError(f"알 수 없는 시나리오: {name} (가능: {list(SCENARIOS)})")
    cfg = dict(SCENARIOS[name])
    cfg["scenario_name"] = name
    return cfg
