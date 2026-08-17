"""
initial_conditions_demo.py
================================================================
모듈식 초기 조건 시스템(initial_conditions) 시연 스크립트.

여러 프리셋(중심발생 / 다중발생 / 무작위발생 / 인구밀도지도 / 사전면역)을
config 만 바꿔가며 생성하고
  · 요약 통계를 출력하고
  · S/I/R/N 초기 히트맵을 result/ 에 저장한다.

마지막에는 만든 초기 조건을 SpatialSIR 모델에 주입해 실제로 돌아가는지
(모델 연결) 짧게 확인한다.
"""

from initial_conditions import (
    create_initial_conditions,
    summarize,
    print_summary,
    plot_initial_conditions,
)

# 각 시나리오: (파일명, 제목, config)
SCENARIOS = [
    (
        "ic_A_center_outbreak.png",
        "A. Central Outbreak (Uniform Population)",
        {
            "grid_size": (120, 120),
            "population_type": "uniform",
            "base_density": 500.0,
            "initial_condition_type": "center_outbreak",
            "initial_infected_fraction": 0.02,
            "outbreak_radius": 4,
        },
    ),
    (
        "ic_B_multiple_outbreaks.png",
        "B. Multiple Outbreaks (User-specified Coordinates)",
        {
            "grid_size": (120, 120),
            "population_type": "uniform",
            "base_density": 500.0,
            "initial_condition_type": "multiple_outbreaks",
            "outbreak_seeds": [
                {"x": 30, "y": 30, "radius": 4, "fraction": 0.03},
                {"x": 90, "y": 40, "radius": 3, "fraction": 0.02},
                {"x": 60, "y": 95, "radius": 5, "fraction": 0.01},
            ],
        },
    ),
    (
        "ic_C_random_outbreaks.png",
        "C. Random Outbreaks (Reproducible, seed=7)",
        {
            "grid_size": (120, 120),
            "population_type": "uniform",
            "base_density": 500.0,
            "initial_condition_type": "random_outbreaks",
            "num_random_seeds": 8,
            "outbreak_radius": 3,
            "initial_infected_fraction": 0.02,
            "random_seed": 7,
        },
    ),
    (
        "ic_D_density_multiple_centers.png",
        "D. Population Density Map (Polycentric Urban) + Central Outbreak",
        {
            "grid_size": (120, 120),
            "population_type": "multiple_centers",
            "population_params": {"base_density": 60.0, "peak_density": 1500.0},
            "initial_condition_type": "center_outbreak",
            "initial_infected_fraction": 0.02,
            "outbreak_radius": 4,
        },
    ),
    (
        "ic_D_density_random_field.png",
        "D. Population Density Map (Random Density Field) + Random Outbreaks",
        {
            "grid_size": (120, 120),
            "population_type": "random_field",
            "population_params": {"base_density": 80.0, "peak_density": 1400.0,
                                  "smoothness": 10},
            "initial_condition_type": "random_outbreaks",
            "num_random_seeds": 6,
            "outbreak_radius": 3,
            "initial_infected_fraction": 0.03,
            "random_seed": 42,
        },
    ),
    (
        "ic_E_prevaccinated.png",
        "E. Pre-immunity/Vaccination (High-density City + 40% Vaccinated)",
        {
            "grid_size": (120, 120),
            "population_type": "high_density_center",
            "population_params": {"base_density": 100.0, "peak_density": 1500.0},
            "initial_condition_type": "center_outbreak",
            "initial_infected_fraction": 0.02,
            "outbreak_radius": 4,
            "recovered_fraction": 0.40,
            "vaccinated_regions": [
                {"x": 60, "y": 60, "radius": 25, "fraction": 0.5},
            ],
        },
    ),
]


def run_demo():
    for fname, title, config in SCENARIOS:
        print("\n" + "=" * 60)
        print(f"[{title}]")
        S, I, R, N = create_initial_conditions(config)
        stats = summarize(S, I, R, N)
        print_summary(stats, title=title)
        plot_initial_conditions(S, I, R, N, title=title, filename=fname)


def demo_feed_into_model():
    """만든 초기 조건을 실제 반응-확산 모델에 주입해 몇 스텝 굴려본다."""
    from sir_reaction_diffusion import SpatialSIR, DX, BETA, GAMMA, D_S, D_I, D_R

    config = {
        "grid_size": (120, 120),
        "population_type": "high_density_center",
        "population_params": {"base_density": 100.0, "peak_density": 1500.0},
        "initial_condition_type": "center_outbreak",
        "initial_infected_fraction": 0.02,
        "outbreak_radius": 4,
    }
    S, I, R, N = create_initial_conditions(config)

    ny, nx = N.shape
    model = SpatialSIR(nx, ny, DX, BETA, GAMMA, D_S, D_I, D_R, N)
    model.set_initial_state(S, I, R, N)

    print("\n" + "=" * 60)
    print("[모델 연결 확인] 초기 조건을 SpatialSIR 에 주입 후 100스텝(10일) 전진")
    print(f"  주입 직후 총 I = {model.totals()[1]:,.0f} 명")
    for _ in range(100):
        model.step(0.1)
    tS, tI, tR = model.totals()
    print(f"  10일 후   총 S/I/R = {tS:,.0f} / {tI:,.0f} / {tR:,.0f} 명")
    print(f"  인구 보존 확인: S+I+R = {tS + tI + tR:,.0f} (초기 N={N.sum():,.0f})")


if __name__ == "__main__":
    run_demo()
    demo_feed_into_model()
    print("\n완료.")
