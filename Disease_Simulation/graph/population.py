"""
population.py
================================================================
Census 2025 county 인구 추계(data/co-est2025-pop.xlsx)를 읽어
county GEOID → population 딕셔너리로 만든다.

엑셀은 county 를 'County Name, State' 로 식별하고 graph 는 GEOID(FIPS)를 쓰므로,
(주 FIPS + 정규화한 county 이름 + 독립시 여부) 로 매칭한다.
버지니아 등 '독립시 vs 동명 카운티'(예: Richmond city / Richmond County) 는
엑셀의 ' city' 접미사와 geojson 의 LSAD=='25' 로 구분한다.
"""

import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POP_XLSX = os.path.join(ROOT, "data", "co-est2025-pop.xlsx")

# 주 이름 → STATEFP(2자리 FIPS)
STATE_NAME_TO_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24",
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28",
    "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49",
    "Vermont": "50", "Virginia": "51", "Washington": "53", "West Virginia": "54",
    "Wisconsin": "55", "Wyoming": "56", "Puerto Rico": "72",
}

# county 이름 뒤에 붙는 행정단위 접미사(긴 것부터 제거)
_SUFFIXES = [" City and Borough", " Census Area", " Municipality", " Borough",
             " Parish", " County", " Municipio", " District", " city"]


def _normalize(name):
    """행정단위 접미사를 떼고 소문자로 정규화한 이름 key."""
    for suf in _SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.strip().lower()


def load_county_population(gdf, path=POP_XLSX, year=2025, verbose=True):
    """
    gdf(index=GEOID, columns 에 STATEFP·NAME·LSAD 포함)에 대응하는
    GEOID → 인구(정수) 딕셔너리를 반환한다.

    반환: (pop_dict, stats) — stats 에 매칭/미매칭 개수.
    """
    df = pd.read_excel(path, sheet_name=0, header=None)

    # 데이터 행: col0 이 '.' 로 시작. 컬럼: 0=area,1=2020base,2=2020,...,7=2025
    year_col = {2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6, 2025: 7}[year]
    rows = df[df[0].astype(str).str.startswith(".")].copy()
    rows["area"] = rows[0].str.lstrip(".")
    rows["county_raw"] = rows["area"].str.rsplit(",", n=1).str[0].str.strip()
    rows["state"] = rows["area"].str.rsplit(",", n=1).str[1].str.strip()

    # 엑셀 쪽 lookup: (STATEFP, name_key, is_city) → population
    pop_lookup = {}
    for _, r in rows.iterrows():
        fips = STATE_NAME_TO_FIPS.get(r["state"])
        if fips is None:
            continue
        is_city = r["county_raw"].endswith(" city")
        key = (fips, _normalize(r["county_raw"]), is_city)
        pop_lookup[key] = int(r[year_col])

    # gdf 각 county 를 같은 key 로 조회
    pop_dict = {}
    unmatched = []
    for geoid, row in gdf.iterrows():
        is_city = (row["LSAD"] == "25")
        key = (row["STATEFP"], _normalize(row["NAME"]), is_city)
        if key in pop_lookup:
            pop_dict[geoid] = pop_lookup[key]
        else:
            unmatched.append((geoid, row["NAME"]))

    stats = {"matched": len(pop_dict), "total": len(gdf), "unmatched": unmatched}
    if verbose:
        print(f"[pop] {year}년 인구 매칭: {len(pop_dict)}/{len(gdf)} county")
        if unmatched:
            print(f"[pop] 미매칭 {len(unmatched)}곳(예): {unmatched[:8]}")
    return pop_dict, stats
