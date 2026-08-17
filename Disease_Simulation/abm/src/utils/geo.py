"""
지도(geojson) 로딩 · 인구 배정 (이식성 있는 자동 처리).

legacy/nyc_metapop_sird_final.py 의 load_geojson / assign_population 을
그대로 옮겨오고, 반복 실행(시나리오·스윕) 대비 캐싱만 추가했다.
"""
import os
import glob
import geopandas as gpd

from src.utils.paths import DATA_DIR, PROJECT_ROOT

TARGET_CRS = "EPSG:32618"          # 미터 투영(거리·면적 계산용)

# (폴백용) 실제 2020 US 센서스 자치구 인구
BOROUGH_POP_2020 = {
    "1": 1_694_251, "2": 1_472_654, "3": 2_736_074,
    "4": 2_405_464, "5": 495_747,
}

_CACHE = {}


def _find_geojson():
    for folder in (DATA_DIR, PROJECT_ROOT):
        m = sorted(glob.glob(os.path.join(folder, "*.geojson")))
        if m:
            return m[0]
    raise FileNotFoundError(f"{DATA_DIR} 에 .geojson 이 없습니다.")


def _load_raw():
    path = _find_geojson()
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(TARGET_CRS)

    if "ntatype" in gdf.columns:          # 주거지(0)만 사용
        gdf = gdf[gdf["ntatype"] == "0"]
    gdf = gdf.reset_index(drop=True)

    name_col = next((c for c in ["ntaname", "name", "NAME", "neighborhood"]
                     if c in gdf.columns), None)
    gdf["_name"] = gdf[name_col] if name_col else [f"node_{i}" for i in range(len(gdf))]
    return gdf, os.path.basename(path)


def _assign_population(gdf):
    """인구 3단 폴백: (1)인구컬럼 (2)센서스×면적 (3)면적비례."""
    gdf = gdf.copy()
    gdf["_area"] = gdf.geometry.area.astype(float)

    pop_col = next((c for c in ["population", "pop", "Pop", "totpop", "pop2020"]
                    if c in gdf.columns), None)
    if pop_col:
        gdf["_pop"] = gdf[pop_col].astype(float).clip(lower=1.0)
        src = f"컬럼 '{pop_col}'"
    elif "borocode" in gdf.columns:
        gdf["_pop"] = 0.0
        for code, pop in BOROUGH_POP_2020.items():
            m = gdf["borocode"] == code
            if m.any():
                a = gdf.loc[m, "_area"]
                gdf.loc[m, "_pop"] = pop * a / a.sum()
        gdf["_pop"] = gdf["_pop"].clip(lower=1.0)
        src = "센서스 자치구 인구 × 면적비례"
    else:
        gdf["_pop"] = gdf["_area"] / gdf["_area"].mean() * 1000.0
        src = "면적 비례(균일 밀도 가정)"
    gdf.attrs["pop_source"] = src
    return gdf


def load_prepared_gdf(verbose=False):
    """지도 로드 + 인구 배정한 GeoDataFrame 반환(프로세스 내 캐싱)."""
    if "gdf" not in _CACHE:
        gdf, fname = _load_raw()
        gdf = _assign_population(gdf)
        _CACHE["gdf"] = gdf
        _CACHE["fname"] = fname
        if verbose:
            print(f"지도 로드: {fname} | 동네 {len(gdf)}개 | 인구: {gdf.attrs['pop_source']}")
    return _CACHE["gdf"].copy()
