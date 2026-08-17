"""
nyc_abm_sir.py
================================================================
실제 뉴욕시(NYC) 지도 위에서 도는 에이전트 기반(ABM) SIR 전염병 모델.

※ 이 파일은 PDE(반응-확산) 모델과 완전히 별개다.
  기존 PDE 코드(sir_reaction_diffusion.py 등)는 전혀 사용하지도 수정하지도 않는다.
  여기서는 미분방정식이 아니라 '개별 에이전트'들이 지도 위를 돌아다니며
  접촉으로 병을 옮기는 방식으로 유행을 시뮬레이션한다.

데이터:
  NYC Open Data 2020 Neighborhood Tabulation Areas(NTA) GeoJSON.
  폴더 안의 .geojson 파일을 자동으로 찾아 읽는다(262개 동네 경계).

사용 라이브러리:
  Mesa 3.x (Agent/Model), Mesa-Geo (GeoAgent/GeoSpace),
  GeoPandas·Shapely(지리 처리), Matplotlib(애니메이션).

설치:
  pip install mesa mesa-geo geopandas shapely matplotlib

실행:
  python3 nyc_abm_sir.py
  → result/nyc_abm_sir.gif  (감염 확산 애니메이션)
  → result/nyc_abm_sir_curve.png  (전체 S/I/R 시계열)

좌표계(CRS) 선택 이유:
  원본은 EPSG:4326(위도·경도, 도 단위)라 거리 계산에 부적합하다.
  요구대로 '미터 단위 거리'를 쓰기 위해 투영좌표계로 재투영한다.
  EPSG:2263(NY Long Island)은 널리 쓰이지만 단위가 '피트'다. 따라서
  infection_radius_m 등을 진짜 '미터'로 다루기 위해 NYC를 덮는 미터 단위
  투영인 EPSG:32618(UTM zone 18N)을 사용한다. (도 단위 거리는 쓰지 않음)
"""

import os
import glob

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import mesa
from mesa_geo import GeoAgent, GeoSpace

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "AppleGothic"    # 한글 라벨용(macOS)
plt.rcParams["axes.unicode_minus"] = False


# ==================================================================
# 조절 파라미터 (파일 상단에 모아 두어 실험하기 쉽게)
# ==================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")          # 결과물 저장 폴더(프로젝트 규칙)
os.makedirs(RESULT_DIR, exist_ok=True)

TARGET_CRS = "EPSG:32618"        # 미터 단위 투영(UTM 18N) — 거리 계산용

# --- 에이전트 표본 ---
TARGET_AGENTS = 220              # 250개 초과 시 이만큼 무작위 표본
MIN_AGENTS, MAX_AGENTS = 180, 250
RANDOM_SEED = 42                 # 재현성용 고정 시드

# --- 감염 규칙 ---
INFECTION_RADIUS_M = 2000.0      # 감염 반경 [m] (NYC 동네 규모 접촉: 1.5~2.5km)
INFECTION_PROBABILITY = 0.30     # 반경 안에 감염자가 있으면 감염될 확률/시간당
INITIAL_INFECTED = 5             # 초기 감염 동네 수(3~8 권장, 쉽게 변경)

# --- 시간 규칙 ---
# 한 스텝 = 1시간. 7일 뒤 회복.
RECOVERY_TIME_STEPS = 7 * 24     # 168 스텝 = 7일

# --- 이동(사람 움직임): 관성 있는 상관 랜덤워크(correlated random walk) ---
# 사람이 자연스럽게 '걷는' 것처럼: 한 번 정한 방향으로 계속 부드럽게 직진하다가
# 가끔씩만 방향을 튼다(관성). 매 스텝 무작위로 흔들면 제자리에서 부르르 떨게 된다.
#
# 좌표계가 '미터'(EPSG:32618)이므로 보폭도 미터로 준다.
# (위경도(도) 기반 예제의 0.00005 같은 값이 아니라, 뉴욕 규모에 맞춘 작은 미터값)
STEP_SIZE_M = 60.0               # 한 스텝(=1시간)에 걷는 보폭 [m] — 예전보다 훨씬 작게
TURN_PROB = 0.05                 # 매 스텝 5% 확률로만 걷는 방향을 새로 튼다(그 외엔 직진)
HOME_RADIUS_M = 500.0            # 이 반경을 벗어나면 부드럽게 집(동네) 쪽으로 방향을 돌린다
# 행동 유형별 이동성 배수(보폭·활동반경에 영향)
BEHAVIOR_SPEED = {
    "low_mobility": 0.4,         # 집콕형: 작은 보폭, 좁은 활동반경
    "normal": 1.0,               # 보통
    "high_mobility": 2.2,        # 활동형: 큰 보폭, 넓은 활동반경 → 접촉 많음
}
BEHAVIOR_WEIGHTS = [0.3, 0.5, 0.2]   # low / normal / high 비율

# --- 애니메이션 ---
HOURS_PER_FRAME = 3              # 한 프레임 = 3시간 → 움직임이 더 매끄럽게 보임
MAX_HOURS = 60 * 24             # 최대 60일까지만(보통 그 전에 유행 종료)
FPS = 8                          # 느리게(눈으로 따라가기 편한 속도)
ANIM_INTERVAL_MS = 120           # FuncAnimation 프레임 간격 [ms] (라이브 표시 속도)

# 상태별 색
STATE_COLORS = {"S": "#2f6fd0", "I": "#d1352c", "R": "#2e8b57"}


# ==================================================================
# 1. 데이터 로딩
# ==================================================================
def load_nta_geojson(folder=HERE):
    """폴더에서 .geojson 을 자동으로 찾아 읽고, 미터 단위 CRS 로 재투영."""
    matches = sorted(glob.glob(os.path.join(folder, "*.geojson")))
    if not matches:
        raise FileNotFoundError(f"{folder} 에서 .geojson 파일을 찾지 못했습니다.")
    path = matches[0]
    print(f"NTA GeoJSON 로드: {os.path.basename(path)}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(TARGET_CRS)          # 위경도(도) → 미터 투영
    print(f"  전체 NTA 수: {len(gdf)}  |  재투영 CRS: {gdf.crs.to_string()}")
    return gdf


def sample_ntas(gdf, seed=RANDOM_SEED):
    """에이전트 수 규칙에 맞춰 NTA 를 선택한다.

      · 250개 초과  → 고정 시드로 약 220개 무작위 표본
      · 180~250개   → 전부 사용
      · 180개 미만  → 전부 사용 + 경고
    """
    n = len(gdf)
    if n > MAX_AGENTS:
        gdf = gdf.sample(n=TARGET_AGENTS, random_state=seed).reset_index(drop=True)
        print(f"  {n}개 > {MAX_AGENTS} → {len(gdf)}개 무작위 표본(seed={seed})")
    elif n < MIN_AGENTS:
        print(f"  [경고] NTA 가 {n}개로 {MIN_AGENTS}개 미만 → 전부 사용")
    else:
        print(f"  {n}개(180~250 범위) → 전부 사용")
    return gdf.reset_index(drop=True)


# ==================================================================
# 2. 에이전트
# ==================================================================
class PersonAgent(GeoAgent):
    """한 동네를 대표하는 사람 에이전트(mesa-geo GeoAgent).

    속성:
      state           : "S"/"I"/"R"
      x, y            : 현재 위치(미터, 투영좌표)
      home_x, home_y  : 소속 NTA 중심(집) — 활동반경 밖으로 나가면 이쪽으로 돌아옴
      heading         : 현재 진행 방향(라디안) — 관성. 가끔만 새로 튼다
      step_size       : 한 스텝(1시간)에 걷는 보폭 [m]
      home_radius     : 집에서 벗어날 수 있는 활동 반경 [m]
      behavior        : 이동성 유형
      infected_since  : 감염된 시각(시간). 회복 시점 계산에 사용
    """

    def __init__(self, model, home_xy, nta_name, behavior):
        # GeoAgent 는 geometry(Point)와 crs 를 요구한다 → 위치를 점으로 표현
        geometry = Point(home_xy)
        super().__init__(model, geometry, model.crs)

        self.home_x, self.home_y = float(home_xy[0]), float(home_xy[1])
        self.x, self.y = self.home_x, self.home_y
        self.nta_name = nta_name
        self.behavior = behavior

        # 관성(heading): 한 번 정한 방향으로 계속 직진하다가 5% 확률로만 튼다
        self.heading = model.np_rng.uniform(0, 2 * np.pi)          # 진행 방향
        self.step_size = STEP_SIZE_M * BEHAVIOR_SPEED[behavior]     # 보폭 [m]
        self.home_radius = HOME_RADIUS_M * BEHAVIOR_SPEED[behavior] # 활동반경 [m]

        self.state = "S"
        self.infected_since = None

    def infect(self, at_hour):
        self.state = "I"
        self.infected_since = at_hour


# ==================================================================
# 3. 모델
# ==================================================================
class NYCSIRModel(mesa.Model):
    """NYC 지도 위 ABM SIR 모델."""

    def __init__(self, initial_infected=INITIAL_INFECTED, seed=RANDOM_SEED):
        super().__init__(seed=seed)
        self.np_rng = np.random.default_rng(seed)   # 재현 가능한 난수원
        self.hour = 0

        # --- 지도 로딩 + 표본 ---
        gdf = sample_ntas(load_nta_geojson(), seed=seed)
        self.nta = gdf                                # 배경 그리기에 사용
        self.crs = gdf.crs
        self.space = GeoSpace(crs=str(gdf.crs), warn_crs_conversion=False)
        self.bounds = gdf.total_bounds               # [minx,miny,maxx,maxy]

        # --- 에이전트 생성 ---
        self.person_agents = self.create_agents(gdf)

        # --- 초기 감염 ---
        self.seed_initial_infections(initial_infected)

        # --- 기록 ---
        self.history = {"hour": [], "S": [], "I": [], "R": []}
        self.record()

    # ----- 에이전트 생성 -----
    def create_agents(self, gdf):
        """NTA 하나당 대표 에이전트 하나를 그 동네 중심 근처에 만든다."""
        centroids = gdf.geometry.centroid           # 미터 투영이라 안전
        behaviors = self.np_rng.choice(
            list(BEHAVIOR_SPEED.keys()), size=len(gdf), p=BEHAVIOR_WEIGHTS)

        agents = []
        for i, (geom, name, beh) in enumerate(
                zip(centroids, gdf["ntaname"], behaviors)):
            # 중심에서 살짝 무작위로 떨어뜨려 점들이 겹치지 않게(동네 안쪽)
            jitter = self.np_rng.normal(0, 200, size=2)   # ±200m
            home = (geom.x + jitter[0], geom.y + jitter[1])
            a = PersonAgent(self, home, name, beh)
            agents.append(a)
        self.space.add_agents(agents)
        print(f"  에이전트 생성: {len(agents)}명 "
              f"(행동유형 분포 low/normal/high = "
              f"{[int((behaviors==k).sum()) for k in BEHAVIOR_SPEED]})")
        return agents

    def seed_initial_infections(self, k):
        """무작위로 k개 동네를 초기 감염으로(재현 가능)."""
        idx = self.np_rng.choice(len(self.person_agents), size=k, replace=False)
        for j in idx:
            self.person_agents[j].infect(at_hour=0)
        names = [self.person_agents[j].nta_name for j in idx]
        print(f"  초기 감염 {k}곳: {', '.join(names)}")

    # ----- 이동 -----
    def move_agents(self):
        """관성 있는 자연스러운 걷기(상관 랜덤워크).

        · 5% 확률로만 방향을 새로 틀고, 나머지는 정한 방향으로 부드럽게 직진.
        · 집 활동반경을 벗어나면 집 쪽으로 방향을 돌린다.
        · NYC 지도 경계에 부딪히면 벽에 튕긴다(반사) → 지도를 벗어날 수 없다."""
        minx, miny, maxx, maxy = self.bounds
        for a in self.person_agents:
            # (1) 관성: 대부분 직진, 5% 확률로만 새 방향 선택
            if self.np_rng.random() < TURN_PROB:
                a.heading = self.np_rng.uniform(0, 2 * np.pi)

            # (2) 집 활동반경을 넘으면 부드럽게 집 방향으로 재조준
            dx, dy = a.home_x - a.x, a.home_y - a.y
            if np.hypot(dx, dy) > a.home_radius:
                a.heading = np.arctan2(dy, dx)

            # (3) 정한 방향으로 한 보폭 직진
            a.x += a.step_size * np.cos(a.heading)
            a.y += a.step_size * np.sin(a.heading)

            # (4) 지도 경계 반사(bounce) — 벽을 넘어가면 안으로 되돌리고 방향을 튕김
            if a.x < minx:
                a.x = minx + (minx - a.x)          # 넘어간 만큼 안으로 반사
                a.heading = np.pi - a.heading       # 좌우 방향 반전
            elif a.x > maxx:
                a.x = maxx - (a.x - maxx)
                a.heading = np.pi - a.heading
            if a.y < miny:
                a.y = miny + (miny - a.y)
                a.heading = -a.heading              # 상하 방향 반전
            elif a.y > maxy:
                a.y = maxy - (a.y - maxy)
                a.heading = -a.heading

            a.geometry = Point(a.x, a.y)   # GeoAgent 위치 갱신

    # ----- 감염 전파 -----
    def spread_infection(self):
        """반경 안에 감염자가 있는 감염가능자를 확률적으로 감염시킨다.

        규칙(요구사항): 감염자와의 거리가 infection_radius_m 이내이면
        그 감염가능자는 매 시간 INFECTION_PROBABILITY(0.30) 로 감염된다.
        거리는 미터 투영좌표에서 계산(도 단위 사용 안 함)."""
        xs = np.array([a.x for a in self.person_agents])
        ys = np.array([a.y for a in self.person_agents])
        states = np.array([a.state for a in self.person_agents])

        inf = np.where(states == "I")[0]
        sus = np.where(states == "S")[0]
        if inf.size == 0 or sus.size == 0:
            return

        # 감염가능자 × 감염자 거리 제곱 행렬 (미터)
        dx = xs[sus][:, None] - xs[inf][None, :]
        dy = ys[sus][:, None] - ys[inf][None, :]
        near = (dx * dx + dy * dy) <= INFECTION_RADIUS_M ** 2
        exposed = near.any(axis=1)                 # 반경 내 감염자 존재 여부

        draws = self.np_rng.random(sus.size)
        newly = sus[exposed & (draws < INFECTION_PROBABILITY)]
        for j in newly:
            self.person_agents[j].infect(at_hour=self.hour)

    # ----- 회복 -----
    def recover_agents(self):
        """감염된 지 RECOVERY_TIME_STEPS(=7일) 지나면 회복(영구)."""
        for a in self.person_agents:
            if a.state == "I" and (self.hour - a.infected_since) >= RECOVERY_TIME_STEPS:
                a.state = "R"

    # ----- 집계 -----
    def get_sir_counts(self):
        c = {"S": 0, "I": 0, "R": 0}
        for a in self.person_agents:
            c[a.state] += 1
        return c

    def record(self):
        c = self.get_sir_counts()
        self.history["hour"].append(self.hour)
        for k in "SIR":
            self.history[k].append(c[k])

    # ----- 한 스텝(=1시간) -----
    def step(self):
        self.move_agents()        # 사람 이동
        self.spread_infection()   # 접촉 감염
        self.hour += 1            # 시계 진행
        self.recover_agents()     # 회복 판정
        self.record()


# ==================================================================
# 4. 시각화 / 애니메이션
# ==================================================================
def _positions_and_colors(model):
    xs = np.array([a.x for a in model.person_agents])
    ys = np.array([a.y for a in model.person_agents])
    cols = [STATE_COLORS[a.state] for a in model.person_agents]
    return xs, ys, cols


def animate_simulation(model, filename="nyc_abm_sir.gif"):
    """모델을 끝까지 굴리며 프레임을 모은 뒤 애니메이션(GIF)으로 저장.

    NTA 경계를 배경으로 깔고, 그 위에 에이전트를 상태별 색 점으로 그린다."""
    # --- 1) 시뮬레이션을 미리 돌려 프레임 기록(감염 종료 시 조기 종료) ---
    print("\n시뮬레이션 실행 중...")
    frames = []
    xs, ys, cols = _positions_and_colors(model)
    frames.append((model.hour, xs, ys, cols, model.get_sir_counts()))
    while model.hour < MAX_HOURS:
        for _ in range(HOURS_PER_FRAME):
            model.step()
        xs, ys, cols = _positions_and_colors(model)
        frames.append((model.hour, xs, ys, cols, model.get_sir_counts()))
        if model.get_sir_counts()["I"] == 0:      # 유행 종료 → 멈춤
            break
    print(f"  총 {model.hour}시간({model.hour/24:.1f}일), "
          f"{len(frames)}프레임 기록")

    # --- 2) 배경(NTA 경계) + 초기 산점도 ---
    fig, ax = plt.subplots(figsize=(9, 9))
    model.nta.boundary.plot(ax=ax, color="0.75", linewidth=0.5, zorder=1)
    ax.set_aspect("equal")                        # 지도 종횡비 유지
    minx, miny, maxx, maxy = model.bounds
    pad = 0.03 * (maxx - minx)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_xlabel("East-West [m, EPSG:32618]")
    ax.set_ylabel("North-South [m, EPSG:32618]")

    h0, x0, y0, c0, cnt0 = frames[0]
    scat = ax.scatter(x0, y0, c=c0, s=22, edgecolors="white",
                      linewidths=0.3, zorder=3)

    # 범례(상태별 색)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=lab,
                          markerfacecolor=STATE_COLORS[s], markersize=9)
               for s, lab in [("S", "Susceptible S"), ("I", "Infected I"),
                               ("R", "Recovered R")]]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)
    title = ax.set_title("")

    def update(i):
        hour, xs, ys, cols, cnt = frames[i]
        scat.set_offsets(np.column_stack([xs, ys]))
        scat.set_color(cols)
        title.set_text(
            f"NYC ABM SIR   t = {hour}h ({hour/24:.1f} days)   "
            f"S={cnt['S']}  I={cnt['I']}  R={cnt['R']}")
        return scat, title

    # interval: 라이브 표시 시 프레임 간격[ms]. 저장 GIF 속도는 fps 로 결정.
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False,
                         interval=ANIM_INTERVAL_MS)
    out = os.path.join(RESULT_DIR, filename)
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"저장: {out}")

    _plot_sir_curve(model)


def _plot_sir_curve(model):
    """전체 S/I/R 시계열 그래프(부가 결과)."""
    h = np.array(model.history["hour"]) / 24.0     # 일 단위
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h, model.history["S"], color=STATE_COLORS["S"], lw=2.2, label="S Susceptible")
    ax.plot(h, model.history["I"], color=STATE_COLORS["I"], lw=2.2, label="I Infected")
    ax.plot(h, model.history["R"], color=STATE_COLORS["R"], lw=2.2, label="R Recovered")
    ax.fill_between(h, 0, model.history["I"], color=STATE_COLORS["I"], alpha=0.12)
    ax.set_title("NYC ABM SIR — Total Agent State Trajectory", fontsize=13)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of agents")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "nyc_abm_sir_curve.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 5. 메인
# ==================================================================
if __name__ == "__main__":
    model = NYCSIRModel()
    animate_simulation(model)
    print("완료.")
