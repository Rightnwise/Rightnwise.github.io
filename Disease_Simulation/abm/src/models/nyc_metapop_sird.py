"""
NYC 메타포퓰레이션 SIRD 모델 (config 기반, 리팩터링본)
================================================================
legacy/nyc_metapop_sird_final.py 의 모델 로직을 그대로 옮기되:
  · 전역 상수 → SimConfig 로 주입
  · 시각화 분리(모델은 데이터만 반환)
  · run_simulation(config) -> SimulationResult
  · 개입(봉쇄/백신/격리효과) 최소 구현 추가

★ baseline 재현 보장:
  개입 파라미터가 모두 기본값(off)이면, 난수(self.rng) 호출 순서가
  legacy 최종본과 완전히 동일하도록 '원본 경로'를 그대로 탄다.
  (개입이 켜질 때만 추가 난수를 쓰는 분기)

SIRD 상태:
  S(감염가능) · I(감염: 잠복기 L + 증상기) · R(회복/면역) · D(사망)
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

import mesa
import mesa_geo as mg

from src.config.default_config import SimConfig, make_config
from src.utils.geo import load_prepared_gdf


# ==================================================================
# 에이전트
# ==================================================================
class NeighborhoodAgent(mg.GeoAgent):
    """고정 공간 노드(동네). 인구·면적·job_index·destination_pdf 보유."""

    def __init__(self, model, geometry, crs, idx, name, population, area):
        super().__init__(model, geometry, crs)
        self.idx = idx
        self.name = name
        self.population = population
        self.area = area
        self.job_index = 0.0
        self.destination_pdf = None


class PersonAgent(mesa.Agent):
    """사람. condition('S'/'I'/'R'/'D'), home/current(동네 idx)."""

    def __init__(self, model, home_idx):
        super().__init__(model)
        self.condition = "S"
        self.home = home_idx
        self.current = home_idx
        self.infected_at = None
        self.recovered_at = None      # 회복 시각(면역 소실 계산용)
        self.ever_infected = False    # 실제 감염 경험(백신 R 과 구분)
        self.vaccinated = False


# ==================================================================
# 모델
# ==================================================================
class MetapopSIRD(mesa.Model):
    def __init__(self, config=None, gdf=None):
        self.cfg = make_config(config)
        super().__init__(seed=self.cfg.random_seed)
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.tick = 0
        self.cumulative_infections = 0    # 감염 이벤트 수(재감염 포함)
        self.unique_infected = 0          # 한 번이라도 감염된 '사람' 수

        # --- 지도 · 인구 ---
        gdf = load_prepared_gdf() if gdf is None else gdf
        self.gdf = gdf
        self.N = len(gdf)
        self.names = list(gdf["_name"])
        self.pop = gdf["_pop"].to_numpy(float)
        self.area = gdf["_area"].to_numpy(float)

        # 노드
        self.space = mg.GeoSpace(crs=str(gdf.crs), warn_crs_conversion=False)
        self.nodes = []
        for i, row in gdf.iterrows():
            na = NeighborhoodAgent(self, row.geometry, gdf.crs, i,
                                   row["_name"], self.pop[i], self.area[i])
            self.nodes.append(na)
        self.space.add_agents(self.nodes)

        # --- 방법 B: 중심지 + job_index + 출퇴근 PDF ---
        self.build_method_b()

        # --- 사람 에이전트(인구 비례) ---
        self.persons = self.create_agents()
        self.home_groups = [[] for _ in range(self.N)]
        for p in self.persons:
            self.home_groups[p.home].append(p)

        # --- 개입: 백신(시작 전 S→R) → 초기 감염 ---
        self.apply_vaccination()
        self.seed_infection()

    # ---------------- 방법 B ----------------
    def build_method_b(self):
        cent = self.gdf.geometry.centroid
        xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
        diff = xy[:, None, :] - xy[None, :, :]
        Dm = np.sqrt((diff ** 2).sum(-1)) / 1000.0
        radius_km = np.sqrt(self.area / np.pi) / 1000.0
        Dm[np.arange(self.N), np.arange(self.N)] = radius_km
        self.centroids = xy

        density = self.pop / self.area
        top = np.where(density >= density.max() * 0.999)[0]
        c = xy[top].mean(axis=0)
        self.center = int(top[np.argmin(((xy[top] - c) ** 2).sum(1))])
        self.density = density

        d_center = np.sqrt(((xy - xy[self.center]) ** 2).sum(1)) / 1000.0
        self.job_index = 1.0 / (1.0 + d_center)

        W = (self.pop[None, :] * self.job_index[None, :]) / (Dm ** self.cfg.distance_decay_alpha)
        self.pdf = W / W.sum(axis=1, keepdims=True)
        for i, na in enumerate(self.nodes):
            na.job_index = float(self.job_index[i])
            na.destination_pdf = self.pdf[i]

    def create_agents(self):
        persons = []
        for idx in range(self.N):
            n = max(1, round(self.pop[idx] / self.cfg.pop_per_agent)) if self.pop[idx] > 0 else 0
            persons.extend(PersonAgent(self, idx) for _ in range(n))
        return persons

    # ---------------- 개입: 백신 ----------------
    def apply_vaccination(self):
        """시작 전 일부 S 를 R(면역)로. rate<=0 이면 아무 것도 안 함(baseline 난수 보존)."""
        rate = self.cfg.vaccination_rate
        if rate <= 0:
            return
        n_vax = int(round(rate * len(self.persons)))
        if n_vax <= 0:
            return
        strategy = self.cfg.vaccination_strategy
        if strategy == "high_density_first":
            # 인구밀도 높은 동네부터(결정론적). 동률은 에이전트 순서.
            order = sorted(range(len(self.persons)),
                           key=lambda i: -self.density[self.persons[i].home])
            chosen = [self.persons[i] for i in order[:n_vax]]
        elif strategy == "random":
            chosen = self.rng.choice(self.persons, size=n_vax, replace=False)
        else:
            raise ValueError(f"알 수 없는 vaccination_strategy: {strategy}")
        for a in chosen:
            a.condition = "R"
            a.vaccinated = True

    def pick_seed_node(self):
        """최초 발생 지점(동네) 선택. 기본 'north' = 북쪽에서 인구 많은 동네."""
        loc = self.cfg.seed_location
        if isinstance(loc, int):
            return int(loc)
        y = self.centroids[:, 1]
        if loc == "north":
            cand = np.where(y >= np.quantile(y, 0.85))[0]     # 북쪽 15%
        elif loc == "south":
            cand = np.where(y <= np.quantile(y, 0.15))[0]
        elif loc == "center":
            return int(self.center)
        else:
            cand = np.arange(self.N)
        # 그 지역에서 '인구가 가장 많은' 실제 동네를 대표점으로
        return int(cand[np.argmax(self.pop[cand])])

    def seed_infection(self):
        """한 지점(동네)에서 소수의 감염자만 발생시켜 거기서부터 퍼지게 한다."""
        k = self.cfg.initial_infected
        node = self.pick_seed_node()
        self.seed_node = node

        pool = [a for a in self.home_groups[node] if a.condition == "S"]
        if len(pool) < k:                       # 부족하면 가까운 동네까지 확장
            order = np.argsort(((self.centroids - self.centroids[node]) ** 2).sum(1))
            pool = []
            for nid in order:
                pool.extend(a for a in self.home_groups[nid] if a.condition == "S")
                if len(pool) >= k:
                    break
        chosen = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        for p in chosen:
            p.condition = "I"
            p.infected_at = 0
            p.ever_infected = True
        self.cumulative_infections = len(chosen)
        self.unique_infected = len(chosen)

    # ---------------- 스텝 규칙 ----------------
    @property
    def phase(self):
        return "Day" if self.tick % 2 == 0 else "Night"

    @property
    def current_day(self):
        return self.tick // self.cfg.ticks_per_day

    def is_incubating(self, a):
        return (a.condition == "I"
                and (self.tick - a.infected_at) < self.cfg.latent_ticks)

    def effective_commute_prob(self):
        p = self.cfg.commute_probability
        if self.cfg.lockdown_day is not None and self.current_day >= self.cfg.lockdown_day:
            p = p * (1.0 - self.cfg.commute_reduction)
        return p

    def commute(self):
        if self.phase != "Day":
            for a in self.persons:                 # 밤: 사망자 제외 귀가
                if a.condition != "D":
                    a.current = a.home
            return

        eff = self.effective_commute_prob()
        iso = self.cfg.isolation_effectiveness
        if eff >= 1.0 and iso >= 1.0:
            # ===== baseline 경로: legacy 최종본과 난수 순서까지 100% 동일 =====
            for hidx, group in enumerate(self.home_groups):
                movers = [a for a in group
                          if a.condition in ("S", "R") or self.is_incubating(a)]
                if movers:
                    dests = self.rng.choice(self.N, size=len(movers), p=self.pdf[hidx])
                    for a, d in zip(movers, dests):
                        a.current = int(d)
                for a in group:
                    if a.condition == "I" and not self.is_incubating(a):
                        a.current = a.home
        else:
            # ===== 개입(봉쇄/격리효과) 반영 일반 경로 =====
            for hidx, group in enumerate(self.home_groups):
                goers = []
                for a in group:
                    if a.condition == "D":
                        continue
                    symptomatic = (a.condition == "I" and not self.is_incubating(a))
                    if symptomatic:
                        # iso 확률로 자가격리 성공 → 집
                        if iso >= 1.0 or self.rng.random() < iso:
                            a.current = a.home
                            continue
                        # 격리 실패 → 아래에서 출근 후보로
                    # S/R/잠복기/격리실패자: eff 확률로 출근
                    if eff >= 1.0 or self.rng.random() < eff:
                        goers.append(a)
                    else:
                        a.current = a.home
                if goers:
                    dests = self.rng.choice(self.N, size=len(goers), p=self.pdf[hidx])
                    for a, d in zip(goers, dests):
                        a.current = int(d)

    def spread_infection(self):
        present = [[] for _ in range(self.N)]
        for a in self.persons:
            if a.condition != "D":
                present[a.current].append(a)
        beta = self.cfg.beta
        for ppl in present:
            n = len(ppl)
            if n == 0:
                continue
            n_inf = sum(1 for q in ppl if q.condition == "I")
            if n_inf == 0:
                continue
            p_inf = 1.0 - np.exp(-beta * n_inf / n)
            for q in ppl:
                if q.condition == "S" and self.rng.random() < p_inf:
                    q.condition = "I"
                    q.infected_at = self.tick
                    self.cumulative_infections += 1
                    if not q.ever_infected:            # 첫 감염인 사람만 unique 카운트
                        q.ever_infected = True
                        self.unique_infected += 1

    def resolve(self):
        rec = self.cfg.recovery_ticks
        fat = self.cfg.fatality_rate
        for p in self.persons:
            if p.condition == "I" and (self.tick - p.infected_at) >= rec:
                if self.rng.random() < fat:
                    p.condition = "D"
                else:
                    p.condition = "R"
                    p.recovered_at = self.tick         # 면역 소실 시점 계산용

    def waning(self):
        """회복 후 immunity_ticks 경과 시 R → S (면역 소실, SIRS).
           None 이면 영구 면역(아무 것도 안 함). 백신 접종자는 대상 아님."""
        imm = self.cfg.immunity_ticks
        if imm is None:
            return
        for p in self.persons:
            if (p.condition == "R" and p.recovered_at is not None
                    and (self.tick - p.recovered_at) >= imm):
                p.condition = "S"                      # 다시 감염 가능
                p.infected_at = None
                p.recovered_at = None

    def step(self):
        self.commute()
        self.spread_infection()
        self.resolve()
        self.waning()
        self.tick += 1

    # ---------------- 집계 ----------------
    def detailed_counts(self):
        S = R = D = L = I = 0
        lt = self.cfg.latent_ticks
        for a in self.persons:
            c = a.condition
            if c == "S":
                S += 1
            elif c == "R":
                R += 1
            elif c == "D":
                D += 1
            else:  # I
                if (self.tick - a.infected_at) < lt:
                    L += 1
                else:
                    I += 1
        return {"S": S, "L": L, "I": I, "R": R, "D": D}

    def node_infection_rates(self):
        Icnt = np.zeros(self.N)
        tot = np.zeros(self.N)
        for a in self.persons:
            if a.condition == "D":
                continue
            tot[a.current] += 1
            if a.condition == "I":
                Icnt[a.current] += 1
        return np.divide(Icnt, tot, out=np.zeros(self.N), where=tot > 0)

    def final_node_frame(self):
        """동네별(home 기준) 최종 발병률(공격률) GeoDataFrame."""
        ever = np.zeros(self.N)
        deaths = np.zeros(self.N)
        tot = np.zeros(self.N)
        for a in self.persons:
            tot[a.home] += 1
            if a.ever_infected:
                ever[a.home] += 1
            if a.condition == "D":
                deaths[a.home] += 1
        gdf = self.gdf[["_name", "geometry"]].copy()
        gdf["population"] = (self.pop / self.cfg.pop_per_agent)
        gdf["agents"] = tot
        gdf["ever_infected"] = ever
        gdf["deaths"] = deaths
        gdf["attack_rate"] = np.divide(ever, tot, out=np.zeros(self.N), where=tot > 0)
        gdf["is_seed"] = np.arange(self.N) == getattr(self, "seed_node", -1)
        gdf = gdf.rename(columns={"_name": "name"})
        return gdf


# ==================================================================
# 결과 컨테이너 + 실행 함수
# ==================================================================
@dataclass
class SimulationResult:
    timeseries: pd.DataFrame              # day 별 S/L/I/R/D/...
    summary: dict                         # 요약 지표
    final_node_states: gpd.GeoDataFrame   # 동네별 최종 공격률(지도용)
    config: dict                          # 사용한 설정
    center_name: str = ""                 # 자동탐색 중심지 이름
    seed_name: str = ""                   # 최초 발생 동네 이름
    final_agent_states: Optional[pd.DataFrame] = None
    spatial_frames: Optional[list] = None   # 애니메이션용: 프레임별 동네 감염률 배열
    frame_days: Optional[list] = None       # 각 프레임의 day


def run_simulation(config=None, gdf=None, record_agents=False,
                   capture_spatial=False, spatial_stride_days=2):
    """config(dict/SimConfig)로 시뮬레이션을 돌리고 SimulationResult 반환.

    모델은 그래프를 그리지 않고 데이터만 반환한다(시각화는 plots.py).
    capture_spatial=True 이면 spatial_stride_days 마다 동네별 감염률을 저장(GIF용)."""
    cfg = make_config(config)
    model = MetapopSIRD(cfg, gdf=gdf)

    rows = []
    frames, frame_days = [], []
    prev_cum_inf = model.cumulative_infections
    prev_deaths = 0

    def snap_spatial(day):
        frames.append(model.node_infection_rates())
        frame_days.append(day)

    def snapshot(day):
        nonlocal prev_cum_inf, prev_deaths
        c = model.detailed_counts()
        cum_inf = model.cumulative_infections
        deaths = c["D"]
        active = c["L"] + c["I"]
        alive = c["S"] + c["L"] + c["I"] + c["R"]
        rows.append({
            "day": day,
            "S": c["S"], "latent_infected": c["L"], "I": c["I"],
            "R": c["R"], "D": c["D"],
            "active_infected": active,
            "new_infections": cum_inf - prev_cum_inf,
            "new_deaths": deaths - prev_deaths,
            "cumulative_infections": cum_inf,
            "cumulative_unique_infected": model.unique_infected,
            "cumulative_deaths": deaths,
            "alive_population": alive,
        })
        prev_cum_inf = cum_inf
        prev_deaths = deaths

    snapshot(0)                                   # 초기 상태(day 0)
    if capture_spatial:
        snap_spatial(0)
    for day in range(1, cfg.simulation_days + 1):
        for _ in range(cfg.ticks_per_day):        # 낮 + 밤
            model.step()
        snapshot(day)
        done = model.detailed_counts()["L"] + model.detailed_counts()["I"] == 0
        if capture_spatial and (day % spatial_stride_days == 0 or done):
            snap_spatial(day)
        if done:
            break

    ts = pd.DataFrame(rows)
    from src.analysis.metrics import compute_summary
    summary = compute_summary(ts, cfg.scenario_name)

    agents_df = None
    if record_agents:
        agents_df = pd.DataFrame([{"home": a.home, "condition": a.condition,
                                   "ever_infected": a.ever_infected}
                                  for a in model.persons])

    return SimulationResult(
        timeseries=ts,
        summary=summary,
        final_node_states=model.final_node_frame(),
        config=cfg.to_dict(),
        center_name=model.names[model.center],
        seed_name=model.names[model.seed_node],
        final_agent_states=agents_df,
        spatial_frames=frames if capture_spatial else None,
        frame_days=frame_days if capture_spatial else None,
    )
