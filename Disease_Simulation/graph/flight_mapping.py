"""
flight_mapping.py
================================================================
항공 이동 데이터(T-100 Domestic Market, DB28DM)를 county 단위로 매핑한다.

입력:
  data/airports.csv                     — OurAirports (공항 코드 + 위경도)
  data/DD.DB28DM.*.zip                  — T-100 Domestic Market (.asc, pipe 구분)
  data/counties.geojson                 — county 폴리곤

산출:
  data/airport_county_map.csv           — 공항코드 → GEOID 매핑(캐시)
  data/flight_county_matrix_YYYYMM.csv  — county쌍 월간 항공 승객(무방향)

핵심:
  · T-100 공항코드는 IATA(ATL)와 FAA local(05A)가 섞여 있어 두 컬럼 모두로 매칭.
  · 공항 위경도 → counties.geojson 공간조인(within) → GEOID.
  · 같은 county 안의 여러 공항(예: Queens=JFK+LGA)은 자동으로 합쳐진다.
  · 화물전용 편(class=G 등)은 pax>0 필터로 제외.

이 파일은 기존 시뮬레이션 코드를 건드리지 않는 '데이터 준비' 전용 모듈이다.

실행:
  python graph/flight_mapping.py                  # 2019-02, 06, 10 매트릭스 생성
  python graph/flight_mapping.py --months 201910
"""

import os
import glob
import zipfile
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

AIRPORTS_CSV = os.path.join(DATA, "airports.csv")
COUNTIES_GEOJSON = os.path.join(DATA, "counties.geojson")
AIRPORT_MAP_CSV = os.path.join(DATA, "airport_county_map.csv")

# T-100 .asc 컬럼 (pipe 구분, 18개)
T100_COLS = [
    "year", "month", "orig", "orig_mkt", "orig_wac", "orig_city",
    "dest", "dest_mkt", "dest_wac", "dest_city", "carrier", "carrier_ent",
    "cgroup", "distance", "class", "pax", "freight", "mail",
]

# 미국 본토(CONUS) 밖 STATEFP — county graph 와 동일 기준
NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}


# ── 1. 공항코드 → county GEOID 매핑 ─────────────────────────────
def build_airport_county_map(recompute=False, verbose=True):
    """
    airports.csv 의 위경도를 counties.geojson 에 공간조인해 공항코드→GEOID 를 만든다.
    IATA / local_code / ident / gps_code 순으로 코드를 등록(먼저 온 것 우선).
    캐시(AIRPORT_MAP_CSV)가 있으면 재사용.
    """
    if os.path.exists(AIRPORT_MAP_CSV) and not recompute:
        df = pd.read_csv(AIRPORT_MAP_CSV, dtype=str)
        if verbose:
            print(f"[flight] 공항→county 캐시 로드: {len(df)} codes")
        return dict(zip(df["code"], df["GEOID"]))

    ap = pd.read_csv(AIRPORTS_CSV, dtype=str)
    ap["lat"] = pd.to_numeric(ap["latitude_deg"], errors="coerce")
    ap["lon"] = pd.to_numeric(ap["longitude_deg"], errors="coerce")
    us = ap[(ap["iso_country"] == "US") & ap["lat"].notna()
            & (ap["type"] != "closed")].copy()

    # 코드 → 좌표 (여러 코드 컬럼을 모두 등록)
    lut = {}
    for col in ["iata_code", "local_code", "ident", "gps_code"]:
        sub = us[[col, "lat", "lon"]].dropna(subset=[col])
        for code, la, lo in sub.itertuples(index=False):
            lut.setdefault(str(code), (la, lo))

    pts = pd.DataFrame([(c, la, lo) for c, (la, lo) in lut.items()],
                       columns=["code", "lat", "lon"])
    gpts = gpd.GeoDataFrame(
        pts, geometry=[Point(x, y) for x, y in zip(pts["lon"], pts["lat"])],
        crs="EPSG:4326")

    cty = gpd.read_file(COUNTIES_GEOJSON)[["GEOID", "NAME", "STATEFP", "geometry"]]
    joined = gpd.sjoin(gpts, cty, how="inner", predicate="within")
    out = joined[["code", "GEOID", "NAME", "STATEFP"]].drop_duplicates("code")
    out.to_csv(AIRPORT_MAP_CSV, index=False)
    if verbose:
        print(f"[flight] 공항→county 매핑 생성: {len(out)} codes → {AIRPORT_MAP_CSV}")
    return dict(zip(out["code"], out["GEOID"]))


# ── 2. T-100 월별 데이터 로드 (zip 에서 바로) ────────────────────
def _zip_for(year, month):
    """해당 연월을 포함하는 zip 중 가장 최신 릴리스를 고른다."""
    target = year * 100 + month
    best = None
    for z in sorted(glob.glob(os.path.join(DATA, "DD.DB28DM.*.zip"))):
        # 파일명: DD.DB28DM.<start>.<end>.REL01.<date>.zip  → 12개월 롤링 윈도우
        parts = os.path.basename(z).split(".")
        start, end = int(parts[2]), int(parts[3])
        if start <= target <= end:
            best = z          # sorted 이므로 마지막이 최신 릴리스
    return best


def load_t100_month(year, month, verbose=True):
    """zip 안의 .asc 를 읽어 해당 연월·승객>0 행만 반환(압축 해제 불필요)."""
    z = _zip_for(year, month)
    if z is None:
        raise FileNotFoundError(f"{year}-{month:02d} 를 포함하는 T-100 zip 이 없습니다.")
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith(".asc")][0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, sep="|", header=None, names=T100_COLS)
    df = df[(df["year"] == year) & (df["month"] == month) & (df["pax"] > 0)].copy()
    if verbose:
        print(f"[flight] {year}-{month:02d}: {len(df)} rows, "
              f"{int(df['pax'].sum()):,} passengers  (from {os.path.basename(z)})")
    return df


# ── 3. county×county 월간 항공 승객 매트릭스 ────────────────────
def build_county_flight_matrix(year, month, conus_only=True, save=True,
                               verbose=True):
    """
    공항쌍 승객 → county쌍(무방향) 승객으로 집계.
    반환: DataFrame[geoid_a, geoid_b, pax_ab, pax_ba, passengers]
          (geoid_a < geoid_b, passengers = 양방향 합)
    같은 county 내부(공항→같은 county) 이동은 제외한다.
    """
    a2c = build_airport_county_map(verbose=verbose)
    df = load_t100_month(year, month, verbose=verbose)
    total_pax = df["pax"].sum()

    df["og"] = df["orig"].map(a2c)
    df["dg"] = df["dest"].map(a2c)
    df = df[df["og"].notna() & df["dg"].notna()]

    if conus_only:
        df = df[~df["og"].str[:2].isin(NON_CONUS_FIPS)
                & ~df["dg"].str[:2].isin(NON_CONUS_FIPS)]
    df = df[df["og"] != df["dg"]]            # 같은 county 내부 이동 제외

    d = df.groupby(["og", "dg"], as_index=False)["pax"].sum()
    # 무방향으로 합치기 (a<b)
    d["a"] = d[["og", "dg"]].min(axis=1)
    d["b"] = d[["og", "dg"]].max(axis=1)
    d["ab"] = (d["og"] == d["a"]) * d["pax"]
    d["ba"] = (d["og"] == d["b"]) * d["pax"]
    out = (d.groupby(["a", "b"], as_index=False)
             .agg(pax_ab=("ab", "sum"), pax_ba=("ba", "sum")))
    out["passengers"] = out["pax_ab"] + out["pax_ba"]
    out = out.rename(columns={"a": "geoid_a", "b": "geoid_b"})
    out = out.sort_values("passengers", ascending=False).reset_index(drop=True)

    if verbose:
        cov = out["passengers"].sum() / total_pax * 100
        n_cty = len(set(out["geoid_a"]) | set(out["geoid_b"]))
        print(f"[flight] county쌍 {len(out):,}개 · 관여 county {n_cty}개 · "
              f"승객 {int(out['passengers'].sum()):,} (원본의 {cov:.1f}%)")

    if save:
        path = os.path.join(DATA, f"flight_county_matrix_{year}{month:02d}.csv")
        out.to_csv(path, index=False)
        if verbose:
            print(f"[flight] 저장: {path}")
    return out


# ── main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="T-100 항공 데이터 → county 매트릭스")
    p.add_argument("--months", nargs="*", default=["201902", "201906", "201910"],
                   help="YYYYMM (기본: 2019-02, 06, 10)")
    p.add_argument("--recompute-map", action="store_true",
                   help="공항→county 매핑 캐시를 무시하고 다시 계산")
    args = p.parse_args()

    build_airport_county_map(recompute=args.recompute_map)
    for ym in args.months:
        y, m = int(ym[:4]), int(ym[4:])
        print(f"\n──────── {y}-{m:02d} ────────")
        build_county_flight_matrix(y, m)


if __name__ == "__main__":
    main()
