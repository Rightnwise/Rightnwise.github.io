"""
flight_coupling.py
================================================================
county graph 에 '항공 이동 layer' 를 더한다(지상 인접 diffusion 은 그대로 유지).

■ 왜 별도 layer 인가
  지상 확산은 절대수 차이 flux: D*w*(X_a - X_b)  → 인접 county 로만, 인구를 평탄화.
  항공은 성격이 다르다: 먼 거리를 '점프'하고, 여행자는 돌아오므로 인구는 안 변한다.

■ 정규화 (T-100 10월 데이터 분석 결과 반영)
  1) catchment 보정: 공항 승객을 반경 R km 내 county 에 '인구 비례'로 배분.
     - 이유: ATL 은 Clayton(29만)에 있지만 실제 상권은 애틀랜타 광역(610만).
       보정 없이 쓰면 Clayton 이 인구의 89%/일 이동 → 발산.
     - R=100km 면 최대 1인당 이동률 0.897 → 0.054/일 (현실적).
  2) 대칭화: P_ij = (P_ij + P_ji)/2.
     - 비대칭 그대로 두면 순이동(P_ab-P_ba)이 매일 누적 → MCO 연 60만명 유입 같은 왜곡.
     - 실제 순불균형은 총이동의 0.73%(편도표·기재 재배치 노이즈)라 대칭화가 타당.
  3) 유병률(prevalence) 기반 flux:
         flux_X = D_air * P_ij * (X_a/N_a - X_b/N_b) * dt
     - 하루 P명이 이동, 그중 감염자 비율이 X/N → 옮겨가는 양 = P·X/N.
     - S+I+R 을 다 더하면 P*(1-1)=0  → county 인구 N 이 '정확히' 보존된다.
  4) D_air 은 무차원 배율: 1.0=실제 이동량, 0=항공 차단, 0.3=여행 70% 감축.

■ 계산 (factored form — county쌍 66만 edge 를 만들지 않는다)
  county쌍으로 전개하면 665,703 edge(지상의 73배)라 느리다. 수학적으로 동일한
  인수분해 형태로 공항 edge(4,238개)에서만 계산한다:
      p_a      = Σ_i w_ia · (X_i/N_i)        (county → 공항, gather)
      inflow_a = Σ_b P_ab · p_b              (공항 edge)
      out_a    = Σ_b P_ab                    (상수, 미리 계산)
      A_i      = Σ_a w_ia · inflow_a         (공항 → county, scatter)
      B_i      = Σ_a w_ia · out_a            (상수, 미리 계산)
      ΔX_i     = D_air · dt · (A_i − (X_i/N_i)·B_i)
  → Σ_X ΔX_i = B_i − B_i = 0  이므로 county 인구 N 이 정확히 보존된다.

실행(진단):
  python graph/flight_coupling.py --month 201910 --radius 100
"""

import os
import zipfile
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

AIRPORTS_CSV = os.path.join(DATA, "airports.csv")
T100_COLS = [
    "year", "month", "orig", "orig_mkt", "orig_wac", "orig_city",
    "dest", "dest_mkt", "dest_wac", "dest_city", "carrier", "carrier_ent",
    "cgroup", "distance", "class", "pax", "freight", "mail",
]
DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


# ── 공항 코드 → 위경도 ───────────────────────────────────────────
def load_airport_coords():
    ap = pd.read_csv(AIRPORTS_CSV, dtype=str)
    ap["lat"] = pd.to_numeric(ap["latitude_deg"], errors="coerce")
    ap["lon"] = pd.to_numeric(ap["longitude_deg"], errors="coerce")
    us = ap[(ap["iso_country"] == "US") & ap["lat"].notna()
            & (ap["type"] != "closed")]
    lut = {}
    for col in ["iata_code", "local_code", "ident", "gps_code"]:
        sub = us[[col, "lat", "lon"]].dropna(subset=[col])
        for code, la, lo in sub.itertuples(index=False):
            lut.setdefault(str(code), (la, lo))
    return lut


# ── T-100 월별 공항쌍 승객 (+ 대칭성 진단·대칭화) ────────────────
def load_airport_pairs(year, month, symmetric=True, verbose=True):
    """
    반환: DataFrame[a, b, pax_day]  (a<b, 무방향, 대칭화된 하루 승객수)
    symmetric=True → P=(P_ab+P_ba)/2 로 대칭화(인구 보존 보장).
    """
    zips = [f for f in os.listdir(DATA) if f.startswith("DD.DB28DM.") and f.endswith(".zip")]
    target = year * 100 + month
    pick = None
    for z in sorted(zips):
        parts = z.split(".")
        if int(parts[2]) <= target <= int(parts[3]):
            pick = z
    if pick is None:
        raise FileNotFoundError(f"{year}-{month:02d} 를 포함하는 T-100 zip 없음")

    with zipfile.ZipFile(os.path.join(DATA, pick)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".asc")][0]
        t = pd.read_csv(zf.open(name), sep="|", header=None, names=T100_COLS)
    t = t[(t["year"] == year) & (t["month"] == month) & (t["pax"] > 0)]

    d = t.groupby(["orig", "dest"], as_index=False)["pax"].sum()
    d["a"] = d[["orig", "dest"]].min(axis=1)
    d["b"] = d[["orig", "dest"]].max(axis=1)
    d["fwd"] = np.where(d["orig"] == d["a"], d["pax"], 0.0)
    d["bwd"] = np.where(d["orig"] == d["b"], d["pax"], 0.0)
    g = d.groupby(["a", "b"], as_index=False).agg(fwd=("fwd", "sum"), bwd=("bwd", "sum"))

    # ── 대칭성 진단 ──
    both = g[(g["fwd"] > 0) & (g["bwd"] > 0)]
    asym = ((both["fwd"] - both["bwd"]).abs()
            / (both["fwd"] + both["bwd"])) if len(both) else pd.Series([0.0])
    oneway = g[(g["fwd"] == 0) | (g["bwd"] == 0)]
    if verbose:
        print(f"[air] {year}-{month:02d} 공항쌍 {len(g):,} "
              f"(양방향 {len(both):,} / 단방향 {len(oneway):,})")
        print(f"[air] 대칭성: 비대칭지수 중앙값 {asym.median():.3f}, "
              f"평균 {asym.mean():.3f} | 단방향 승객비중 "
              f"{oneway[['fwd','bwd']].sum().sum()/g[['fwd','bwd']].sum().sum()*100:.3f}%")

    if symmetric:
        g["pax_month"] = (g["fwd"] + g["bwd"]) / 2.0   # 양방향 평균 → 순이동 0
    else:
        g["pax_month"] = g["fwd"] + g["bwd"]           # (비권장: 인구 보존 깨짐)

    days = DAYS_IN_MONTH[month]
    g["pax_day"] = g["pax_month"] / days
    if verbose:
        print(f"[air] 대칭화 P=(Pab+Pba)/2 적용 → 순이동 0 (인구 보존). "
              f"총 {g['pax_day'].sum()*2:,.0f} 명/일 이동")
    return g[["a", "b", "pax_day"]]


# ── catchment: 공항 → 반경 내 county 인구비례 가중치 ─────────────
def build_airport_catchment(gdf, codes, radius_km=100.0, populations=None,
                            verbose=True):
    """
    gdf: EPSG:5070, index=GEOID, 'cx','cy' 보유 (compute_centroids 결과)
    반환: (airport_list, apt_idx, cnty_idx, w) — 희소 catchment 행렬
          Σ_i w[i for airport a] = 1
    """
    lut = load_airport_coords()
    codes = [c for c in codes if c in lut]
    cx = gdf["cx"].to_numpy()
    cy = gdf["cy"].to_numpy()
    if populations is None:
        popv = np.full(len(gdf), 1.0)
    else:
        popv = np.array([float(populations.get(g, 0.0)) for g in gdf.index])

    pts = gpd.GeoSeries([Point(lut[c][1], lut[c][0]) for c in codes],
                        crs="EPSG:4326").to_crs(5070)
    px = np.array([p.x for p in pts]), np.array([p.y for p in pts])

    apt_idx, cnty_idx, w = [], [], []
    kept = []
    for k, (x, y) in enumerate(zip(px[0], px[1])):
        dist = np.hypot(cx - x, cy - y) / 1000.0
        m = np.where(dist <= radius_km)[0]
        if m.size == 0 or popv[m].sum() <= 0:
            continue
        ww = popv[m] / popv[m].sum()
        a = len(kept)
        kept.append(codes[k])
        apt_idx.extend([a] * m.size)
        cnty_idx.extend(m.tolist())
        w.extend(ww.tolist())

    if verbose:
        print(f"[air] catchment(R={radius_km:.0f}km): 공항 {len(kept)}개, "
              f"공항당 평균 county {len(cnty_idx)/max(len(kept),1):.1f}개, "
              f"닿는 county {len(set(cnty_idx))}개")
    return (kept, np.array(apt_idx, dtype=int),
            np.array(cnty_idx, dtype=int), np.array(w, dtype=float))


# ── 항공 operator (factored) ─────────────────────────────────────
def build_flight_operator(gdf, year=2019, month=10, radius_km=100.0,
                          populations=None, D_air=1.0, symmetric=True,
                          stochastic=False, seed=0, verbose=True):
    """
    시뮬레이션 loop 에서 쓸 factored 항공 연산자를 만든다(한 번만 계산).
    반환 dict: apt_idx, cnty_idx, w, eu, ev, P, out, B, D_air, n_apt, n_cty, rng

    stochastic=True 면 감염자 유입을 Poisson 정수로 뽑는다(확률적 유입).
    seed 로 재현 가능. 결과가 매번 달라지므로 여러 번(앙상블) 돌려서 봐야 한다.
    """
    pairs = load_airport_pairs(year, month, symmetric=symmetric, verbose=verbose)
    codes = sorted(set(pairs["a"]) | set(pairs["b"]))
    kept, apt_idx, cnty_idx, w = build_airport_catchment(
        gdf, codes, radius_km=radius_km, populations=populations, verbose=verbose)

    pos = {c: i for i, c in enumerate(kept)}
    p = pairs[pairs["a"].isin(pos) & pairs["b"].isin(pos)]
    eu = p["a"].map(pos).to_numpy()
    ev = p["b"].map(pos).to_numpy()
    P = p["pax_day"].to_numpy(dtype=float)

    n_apt, n_cty = len(kept), len(gdf)
    out = np.zeros(n_apt)                      # 공항별 총 이동량(상수)
    np.add.at(out, eu, P)
    np.add.at(out, ev, P)
    B = np.zeros(n_cty)                        # county 별 총 이동량(상수)
    np.add.at(B, cnty_idx, w * out[apt_idx])

    if verbose:
        popv = np.array([float((populations or {}).get(g, 1.0)) for g in gdf.index])
        rate = np.divide(B, popv, out=np.zeros_like(B), where=popv > 0)
        print(f"[air] 공항 edge {len(P):,}개 (county쌍 전개 대신 factored)")
        print(f"[air] 1인당 일간 항공률: 중앙값 {np.median(rate[rate>0]):.5f}, "
              f"최대 {rate.max():.5f}  → CFL {rate.max()*0.05:.5f} (≪1 안정)")
    if verbose and stochastic:
        print(f"[air] 확률적 유입 ON (Poisson 정수 감염자, seed={seed})")
    return {"apt_idx": apt_idx, "cnty_idx": cnty_idx, "w": w,
            "eu": eu, "ev": ev, "P": P, "out": out, "B": B,
            "D_air": D_air, "n_apt": n_apt, "n_cty": n_cty,
            "rng": (np.random.default_rng(seed) if stochastic else None)}


# ── 한 timestep 항공 flux 적용 (S/I/R) ───────────────────────────
def apply_flight_flux(S, I, R, N, flight, dt):
    """
    유병률 기반 항공 이동을 factored 형태로 한 step 적용.
    ΔX_i = D_air·dt·( A_i − (X_i/N_i)·B_i ),  A_i = Σ_a w_ia·Σ_b P_ab·p_b
    Σ_X ΔX = 0 이므로 county 인구 N 은 정확히 보존된다.

    flight["rng"] 가 있으면 '확률적 유입(stochastic importation)' 모드:
      · 감염자 이동 수를 Poisson 으로 뽑아 '정수' 로만 옮긴다(0.001명 유입 제거).
      · 좌석 보존: 총 여행자 수 T 는 그대로 두고 감염자만 정수화한 뒤,
        그 차이를 S 로 보정 → county 인구 N 은 여전히 정확히 보존된다.
    """
    D = flight["D_air"]
    if D == 0.0 or len(flight["P"]) == 0:
        return S, I, R
    ai, ci, w = flight["apt_idx"], flight["cnty_idx"], flight["w"]
    eu, ev, P, B = flight["eu"], flight["ev"], flight["P"], flight["B"]
    n_apt = flight["n_apt"]
    rng = flight.get("rng")
    safeN = np.where(N > 0, N, 1.0)

    # ── 결정론적 flux (기존) ──
    prev, det = {}, {}
    for key, X in (("S", S), ("I", I), ("R", R)):
        sX = np.where(N > 0, X / safeN, 0.0)          # 유병률
        p = np.zeros(n_apt)
        np.add.at(p, ai, w * sX[ci])                  # ① county → 공항
        inflow = np.zeros(n_apt)
        np.add.at(inflow, eu, P * p[ev])              # ② 공항 edge
        np.add.at(inflow, ev, P * p[eu])
        A = np.zeros(len(X))
        np.add.at(A, ci, w * inflow[ai])              # ③ 공항 → county
        prev[key] = (sX, p)
        det[key] = D * dt * (A - sX * B)

    if rng is None:
        return S + det["S"], I + det["I"], R + det["R"]

    # ── 확률적 유입: 감염자만 정수화 ──
    sI, pI = prev["I"]
    T = D * P * dt                                    # edge 별 여행자 수(좌석)
    n_uv = rng.poisson(np.minimum(T * pI[eu], T))     # u→v 로 이동한 '정수' 감염자
    n_vu = rng.poisson(np.minimum(T * pI[ev], T))     # v→u

    arrI = np.zeros(n_apt); depI = np.zeros(n_apt)
    np.add.at(arrI, ev, n_uv); np.add.at(depI, eu, n_uv)
    np.add.at(arrI, eu, n_vu); np.add.at(depI, ev, n_vu)

    A_arr = np.zeros(len(I))                          # 도착: 인구비례로 분배
    np.add.at(A_arr, ci, w * arrI[ai])
    safe_pI = np.where(pI > 0, pI, 1.0)               # 출발: county 의 감염자 기여도 비례
    A_dep = np.zeros(len(I))
    np.add.at(A_dep, ci, (w * sI[ci] / safe_pI[ai]) * depI[ai])
    A_dep = np.minimum(A_dep, I)                      # 보유 감염자보다 많이 못 뺀다

    dI = A_arr - A_dep
    # 좌석 보존: 감염자가 예상보다 더(덜) 탔으면 그만큼 S 가 덜(더) 탄 것으로 보정
    dS = det["S"] + (det["I"] - dI)
    return S + dS, I + dI, R + det["R"]


# ── 진단 CLI ─────────────────────────────────────────────────────
def main():
    import sys
    sys.path.insert(0, HERE)
    from county_graph import load_counties, compute_centroids
    from population import load_county_population

    p = argparse.ArgumentParser(description="항공 coupling 진단")
    p.add_argument("--month", default="201910")
    p.add_argument("--radius", type=float, default=100.0)
    args = p.parse_args()
    y, m = int(args.month[:4]), int(args.month[4:])

    gdf = compute_centroids(load_counties(state=None))
    pop, _ = load_county_population(gdf, verbose=False)
    op = build_flight_operator(gdf, y, m, radius_km=args.radius, populations=pop)

    # 보존 검증: 임의 S/I/R 로 한 step 돌려 총량·N 이 유지되는지
    rng = np.random.default_rng(0)
    N = np.array([float(pop.get(g, 10000)) for g in gdf.index])
    I = N * rng.random(len(N)) * 0.05
    R = N * rng.random(len(N)) * 0.10
    S = N - I - R
    S2, I2, R2 = apply_flight_flux(S, I, R, N, op, dt=0.05)
    print("\n=== 한 step 보존 검증 ===")
    print(f"  총 S 변화: {S2.sum()-S.sum():+.3e}")
    print(f"  총 I 변화: {I2.sum()-I.sum():+.3e}")
    print(f"  총 R 변화: {R2.sum()-R.sum():+.3e}")
    print(f"  county N 최대 변화: {np.abs((S2+I2+R2)-N).max():.3e}  (0 이어야 함)")
    print(f"  음수 발생: {min(S2.min(), I2.min(), R2.min()) < -1e-9}")


if __name__ == "__main__":
    main()
