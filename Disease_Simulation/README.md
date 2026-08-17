# NYC 메타포퓰레이션 SIRD 실험 프레임워크

뉴욕시 동(Neighborhood) 단위 **네트워크 기반 메타포퓰레이션 SIRD** 전염병
시뮬레이션과, 파라미터/개입을 바꿔가며 결과를 관찰하는 **실험 도구**.

기존 최종 모델(`legacy/nyc_metapop_sird_final.py`)의 로직을 보존한 채 config
기반으로 재구성했으며, **baseline은 legacy와 매 틱 SIRD 값이 완전히 일치**함(정확 재현).

---

## 폴더 구조

```
Disease_Simulation/
├── pde/                             # ── 반응-확산 PDE 모델(격자 기반) ──
│   ├── pde_smallpox_seird.py        # 천연두 SEIQR + 명시적 격리(Q)
│   ├── pde_periodic_outbreak.py     # SIRS + 주기적 outbreak(동심원 파동)
│   ├── pde_recurrent_sir.py         # 재발 SIR
│   └── pde_polio_*.py               # 소아마비 백신 전략(4종)
├── abm/                             # ── 에이전트/메타포퓰레이션 ABM ──
│   ├── run_baseline.py
│   ├── run_scenario.py
│   └── src/
│       ├── models/nyc_metapop_sird.py   # 모델 + run_simulation(config)->SimulationResult
│       ├── analysis/                    # metrics · scenario_runner · parameter_sweep
│       ├── visualization/plots.py       # 모든 그래프 저장(모델과 분리)
│       ├── config/                      # default_config.py · scenarios/
│       └── utils/                       # paths.py, geo.py(지도 로딩·인구 배정)
├── graph/                           # ── 카운티 그래프 기반 확산 모델 ──
│   ├── county_graph.py              # 카운티 인접 그래프 구성
│   ├── local_sir.py                 # 카운티 단위 국소 SIR
│   ├── graph_coupled_sir.py         # 그래프 결합 SIR
│   ├── reaction_diffusion_sir.py    # 카운티 반응-확산 SIR / SIRS
│   ├── weighted_reaction_diffusion.py # 항공 이동량 가중 반응-확산
│   ├── flight_mapping.py            # 공항→카운티 매핑
│   ├── flight_coupling.py           # 항공 OD 행렬 → 결합 가중치
│   ├── ensemble.py                  # 다중 시드 앙상블(도착시각 분포)
│   ├── population.py · sir_init.py · numpy_backend.py
│   └── app.py                       # Streamlit 대시보드
├── data/                            # 입력 데이터(geojson·인구·항공 OD)
├── result/                          # 실행 시 자동 생성(저장소에는 미포함)
├── legacy/                          # 이전 스크립트 백업(삭제 안 함)
├── lotka_volterra_allee.py          # 로트카-볼테라(포식-피식) 위상궤도
├── parameter_explorer.py            # ABM 파라미터 탐색 도구
└── README.md
```

---

## 실행 파일 역할 & 방법

| 명령어 | 역할 | 결과 위치 |
|---|---|---|
| `python abm/run_baseline.py` | baseline 1회 실행(legacy와 동일) | `result/baseline/` |
| `python abm/run_scenario.py --scenario <이름>` | 단일 시나리오 실행 | `result/<이름>/` |
| `python abm/src/analysis/scenario_runner.py` | 9개 시나리오 전부 + 비교그래프 | `result/<각>/`, `result/comparison/` |
| `python abm/src/analysis/parameter_sweep.py` | 민감도 분석(β·이동감소·잠복기) + 히트맵 | `result/sensitivity/` |
| `python parameter_explorer.py [--옵션] [--animate]` | 파라미터 직접 조정 1회 실행(CLI). `--animate` 시 GIF 생성 | `result/explorer/` |
| `python abm/run_scenario.py --scenario <이름> --gif` | 시나리오 실행 + 감염 확산 GIF | `result/<이름>/animation.gif` |
| `python pde/pde_smallpox_seird.py` (등 `pde/*.py`) | 반응-확산 PDE 모델 실행 | `result/pde_*` |
| `streamlit run parameter_explorer.py` | 대시보드(선택). 사이드바 체크박스로 GIF 표시 | 화면 |

> **감염 확산 애니메이션(GIF)**: 기본 실행은 데이터/정적 그래프만 만든다(빠름).
> `--animate`/`--gif` 또는 Streamlit 체크박스를 켜면 동네별 감염률 시공간 확산을
> GIF(`animation.gif`)로 렌더링한다. Streamlit에서는 `st.image` 로 바로 보인다.

### 모델 API
```python
from src.models.nyc_metapop_sird import run_simulation
result = run_simulation({"beta": 0.12, "lockdown_day": 20, "commute_reduction": 0.7})
result.timeseries        # DataFrame (일별)
result.summary           # dict (요약 지표)
result.final_node_states # GeoDataFrame (동네별 최종 발병률, 지도용)
result.config            # 사용한 설정
```

---

## 조정 가능한 파라미터 (config)

| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `beta` | 0.10 | 감염 계수 | ✅ 작동 |
| `latent_days` | 4 | 잠복기(무증상 전파) 일수 | ✅ 작동 |
| `recovery_days` | 14 | 감염→회복/사망 일수 | ✅ 작동 |
| `immunity_days` | 60 | 회복 후 면역 지속 일수(이후 다시 S). None=영구 | ✅ 작동(신규, SIRS) |
| `fatality_rate` | 0.02 | 치사율 | ✅ 작동 |
| `initial_infected` | 5 | 초기 감염자(소수, 한 지점) | ✅ 작동 |
| `seed_location` | "north" | 최초 발생 지점: north/south/center/노드idx | ✅ 작동(신규) |
| `simulation_days` | 300 | 최대 일수 | ✅ 작동 |
| `commute_probability` | 1.0 | 낮 출근 기본 확률 | ✅ 작동 |
| `commute_reduction` | 0.0 | 봉쇄 시 이동 감소율 | ✅ 작동 |
| `lockdown_day` | None | 봉쇄 시작일(None=없음) | ✅ 작동(신규) |
| `vaccination_rate` | 0.0 | 시작 전 S→R 접종 비율 | ✅ 작동(신규) |
| `vaccination_strategy` | random | `random` / `high_density_first` | ✅ 작동(신규) |
| `isolation_effectiveness` | 1.0 | 증상자 자가격리 성공률 | ✅ 작동(신규) |
| `distance_decay_alpha` | 4.0 | 통근 거리감쇠(↑=국소적→동심원 확산, ↓=허브매개 전역확산) | ✅ 작동 |
| `random_seed` | 42 | 재현성 | ✅ 작동 |

> 시간 단위: 1일 = 2틱(낮/밤). 낮=확률적 출퇴근(중력모형 PDF), 밤=귀가.
> 잠복기 동안은 무증상이라 **정상 출근하며 전염**, 증상 발현 후 자가격리.

---

## 현재 구현된 기능

- **방법 B 출퇴근**: 인구밀도 최고 동네 자동 탐색 → `job_index=1/(1+거리)` →
  중력모형 PDF(`pop·job/거리^α`)로 확률적 출퇴근 자동 생성(외부 데이터 불필요).
- **단일 지점 발생**: 초기 감염은 도시 전역에 흩뿌리지 않고, `seed_location`(기본
  북쪽에서 인구 많은 동네)에서 소수(`initial_infected`)만 발생해 거기서부터 퍼진다.
  발생 지점은 지도/애니메이션에 ★로 표시.
- **SIRD + 잠복기(L) + 면역소실(SIRS)**: S / L(무증상 전파) / I(증상·자가격리) / R / D.
  회복 후 `immunity_days`(기본 60일) 지나면 R→S 로 다시 감염 가능(백신 접종자는 제외).
  ※ 기본 60일이면 면역이 유행 중 풀려 **재유행(endemic) 다중 파동**이 나타난다(유행이
  simulation_days 까지 지속). 90일 이상으로 늘리면 유행이 면역소실을 앞질러 단일 파동으로 끝난다.
  재감염 때문에 '고유 감염자 수(total_infected)'와 '감염 이벤트 수
  (total_infection_events)'를 구분해 집계한다.
- **밀도 의존 감염** `P = 1 - exp(-β·I/N)`, N = 사망자 제외 실시간 생존 인원.
- **개입(최소 구현)**:
  - `lockdown_day` + `commute_reduction`: 지정일부터 출근확률을 `×(1-reduction)`.
  - `vaccination_rate` + `vaccination_strategy`: 시작 전 일부 S→R (`random` /
    `high_density_first`).
  - `isolation_effectiveness`: 증상자가 격리에 성공할 확률(1.0=항상 격리).
- **실제 인구 기반 배치**: 2020 센서스 자치구 인구를 면적 비례로 분배(약 8,800 에이전트).
- **이식성**: geojson 자동 탐색, 인구 컬럼 자동 감지(없으면 센서스→면적 폴백),
  다른 도시 지도로 교체 가능.

---

## 아직 단순하거나 TODO인 부분

- **isolation_effectiveness**: 구현됨(확률적 격리 성공). 다만 격리 실패자는 그날
  평소처럼 출근하는 단순 처리(부분 격리·접촉 감소는 미구현).
- **vaccination_strategy**: `random` / `high_density_first` 두 가지만. 나이별·직업별
  전략은 **미구현(TODO)**.
- **lockdown**: 전 지역 일괄 이동 감소만. 지역별/단계적 봉쇄는 미구현.
- **백신 효능**: 100% 완전면역 가정(부분 효능·시간지연 미구현).
- **재감염/면역소실 없음**: R·백신은 영구 면역.
- baseline 재현을 위해 개입 off 시에는 legacy와 동일한 난수 경로를 사용.

---

## 생성되는 결과 파일

**단일 실행** `result/<이름>/`
- `timeseries.csv` — day, S, latent_infected, I, R, D, active_infected,
  new_infections, new_deaths, cumulative_infections, cumulative_deaths, alive_population
- `summary.csv` — scenario_name, total_infected, total_deaths, peak_infected,
  peak_day, final_attack_rate, final_fatality_rate, epidemic_end_day,
  max_daily_new_infections
- `epidemic_curve.png`, `cumulative_deaths.png`, `final_map.png`

**시나리오 비교** `result/comparison/`
- `active_infected.png`, `cumulative_deaths.png`, `new_infections.png`
- `peak_infected_bar.png`, `total_deaths_bar.png`, `final_attack_rate_bar.png`
- `all_summary.csv`

**민감도 분석** `result/sensitivity/`
- `sensitivity_results.csv`, `heatmap_data.csv`
- `beta_vs_peak_infected.png`, `beta_vs_total_deaths.png`
- `commute_reduction_vs_total_infected.png`, `heatmap_beta_commute_vs_deaths.png`

---

## legacy/

이전 단계 모델(ODE·PDE·좌표 ABM·네트워크 등)과 직전 최종본
(`nyc_metapop_sird_final.py`)은 삭제하지 않고 `legacy/`에 백업.

> legacy 스크립트는 NTA geojson을 참조한다. 예전에는 `legacy/` 안에 사본을 두었으나
> `data/2020_Neighborhood_Tabulation_Areas_(NTAs)_20260706.geojson` 과 완전히 동일해
> 중복 사본은 제거했다. 실행 시 `data/` 의 파일을 가리키도록 경로만 맞추면 된다.

---

## 저장소 구성 정책 (결과물 · 데이터)

### 결과물은 저장소에 포함하지 않는다
`result/` 이하 **모든 산출물(PNG·GIF·CSV)은 저장소에 두지 않는다.** 전부 코드
재실행으로 재생성되는 파생물이고, 특히 애니메이션 GIF는 수백 MB에 달해 저장소를
무겁게 만들기 때문이다. `result/` 는 `.gitignore` 대상이며, 아래 명령을 실행하면
디렉토리가 자동으로 만들어진다.

```bash
# ABM
python abm/run_baseline.py                          # -> result/baseline/
python abm/src/analysis/scenario_runner.py          # -> result/<시나리오>/, result/comparison/
python abm/src/analysis/parameter_sweep.py          # -> result/sensitivity/

# PDE (격자 반응-확산)
python pde/pde_smallpox_seird.py                    # -> result/pde_smallpox_seird*
python pde/pde_periodic_outbreak.py
python pde/pde_recurrent_sir.py
python pde/pde_polio_uniform_vax.py                 # 소아마비 백신 전략 4종
python pde/pde_polio_saiv.py
python pde/pde_polio_sarv.py
python pde/pde_polio_vax_strategy.py

# 카운티 그래프
python graph/weighted_reaction_diffusion.py         # -> result/county_graph/
python graph/ensemble.py                            # -> result/ensemble/
```

각 실행이 만드는 파일 목록은 위의 [생성되는 결과 파일](#생성되는-결과-파일) 절을 참고.

### 커밋하지 않는 입력 데이터

`data/airports.csv` (12MB, OurAirports 공항 좌표) **하나만** `.gitignore` 대상이다.
공개 URL에서 한 줄로 받을 수 있기 때문이며, 항공 결합 기능을 쓰려면 먼저 받아야 한다.

```bash
curl -L -o data/airports.csv https://ourairports.com/data/airports.csv
```

T-100 원본(`data/DD.DB28DM.*.zip`, 3개 합계 7.6MB)은 US DOT BTS TranStats 폼
다운로드라 재취득이 번거로워 **저장소에 포함**했다.

**airports.csv 없이 되는 것 / 안 되는 것**

| | 필요 여부 |
|---|---|
| `pde/*.py`, `abm/*`, `parameter_explorer.py` | 불필요 — 그대로 실행 |
| `graph/` 지상 확산 모델(`local_sir`·`graph_coupled_sir`·`reaction_diffusion_sir`, `weighted_reaction_diffusion` 을 `--flight-month` 없이) | 불필요 |
| `weighted_reaction_diffusion.py --flight-month YYYYMM`, `graph/ensemble.py`, `graph/app.py` 의 항공 결합 | **필요** — `flight_coupling.py` 가 `airports.csv` 와 T-100 zip 을 직접 읽으며 캐시 폴백이 없다 |

> `data/airport_county_map.csv` · `flight_county_matrix_*.csv` 는 가공 캐시로 저장소에
> 포함돼 있지만, `flight_coupling.build_flight_operator()` 는 이 캐시를 쓰지 않고
> 원본을 다시 읽는다. 따라서 항공 가중 모델을 돌리려면 위 `curl` 이 필수다.

### 저장소에 두지 않는 것 요약

| 대상 | 이유 |
|---|---|
| `result/` 전체 | 코드 재실행으로 재생성되는 파생물 (GIF 포함 시 수백 MB) |
| `__pycache__/`, `*.pyc` | 빌드 산출물 |
| `.DS_Store` | macOS Finder 메타데이터 |
| `data/airports.csv` | 12MB, 공개 URL에서 `curl` 한 줄로 재취득 |

> `legacy/` 는 이전 단계 **스크립트**만 백업으로 남기고, 거기서 나왔던 결과 이미지는
> 함께 정리했다. 필요하면 해당 스크립트를 다시 실행하면 된다.
