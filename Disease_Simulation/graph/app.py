"""
app.py — County Graph Disease Simulation Explorer (Streamlit)
================================================================
기존 county graph reaction-diffusion SIR/SIRS 모델을 웹 UI 에서 실험하는 앱.

원칙:
  · 기존 simulation core 함수를 import 해서 그대로 재사용(모델식·backend·weight 불변).
  · slider 를 바꿔도 자동 실행 X. 반드시 [Run Simulation] 버튼을 눌러야 실행.
  · 무거운 로딩(GeoJSON·graph·edge weight)은 st.cache_data 로 캐시.
  · 시뮬레이션은 빠른 NumPy backend 사용.

실행:  streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from time import perf_counter

# app.py 가 graph/ 안에 있든 프로젝트 루트에 있든 동작하도록 경로 해석
HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = HERE if os.path.basename(HERE) == "graph" else os.path.join(HERE, "graph")
ROOT = os.path.dirname(GRAPH_DIR)          # Disease_Simulation/
sys.path.insert(0, GRAPH_DIR)

from county_graph import load_counties, compute_centroids, build_adjacency_graph
from sir_init import initialize_sir_compartments
from population import load_county_population
from weighted_reaction_diffusion import (
    compute_edge_weights_border_distance, edge_weight_cache_path,
)
from numpy_backend import (
    run_reaction_diffusion_simulation_numpy, numpy_state_to_graph,
)
from reaction_diffusion_sir import (
    plot_timeseries, plot_infected_counties, plot_population_map,
)
from local_sir import plot_final_map
from flight_coupling import build_flight_operator

STREAMLIT_DIR = os.path.join(ROOT, "result", "streamlit")
os.makedirs(STREAMLIT_DIR, exist_ok=True)


# ── 상태 문자열 → load_counties 인자 (US/all/conus → None) ──────
def resolve_state(label):
    return None if str(label).lower() in {"us", "all", "conus"} else label


# ── hist["frames"] → Leaflet 프레임 데이터 [{day, values:{geoid:ratio}}] ──
def build_frames_from_hist(hist, geoid_order):
    """
    frame 의 I/N 배열(gdf.index 순서)을 {geoid: ratio} dict 로 변환.
    JSON 크기를 줄이려고 0 은 생략하고 5자리로 반올림(브라우저에서 없으면 0 처리).
    """
    frames = []
    for f in hist["frames"]:
        vals = {}
        for geoid, r in zip(geoid_order, f["ratio"]):
            if r > 1e-9:
                vals[geoid] = round(float(r), 5)
        frames.append({"day": round(float(f["t"]), 1), "values": vals})
    return frames


# ── Leaflet 인터랙티브 지도 애니메이션 HTML 생성 ─────────────────
def build_leaflet_animation_html(gdf, frames, title="Simulation Map"):
    """
    브라우저에서 county polygon 색상만 갱신하는 Leaflet 애니메이션 HTML 을 반환.
    GeoJSON·frames 를 HTML 안에 JS 변수로 직접 삽입(외부 fetch 없음).
    gdf 는 EPSG:4326, GEOID·NAME 컬럼 포함이어야 한다.
    """
    geoid_col = "GEOID" if "GEOID" in gdf.columns else "geoid"
    name_col = "NAME" if "NAME" in gdf.columns else "name"
    slim = gdf[[geoid_col, name_col, "geometry"]].rename(
        columns={geoid_col: "GEOID", name_col: "NAME"})
    geojson_str = slim.to_json()
    frames_str = json.dumps(frames)

    tmpl = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;font-family:sans-serif}
  #map{height:560px;width:100%}
  #controls{display:flex;align-items:center;gap:12px;padding:8px 10px;
    background:#fafafa;border-bottom:1px solid #ddd;flex-wrap:wrap}
  #controls button{padding:4px 12px;cursor:pointer}
  #frame-slider{flex:1;min-width:160px}
  #day-label{font-weight:bold;min-width:90px}
  .legend{background:#fff;padding:6px 8px;line-height:18px;color:#333;
    box-shadow:0 0 6px rgba(0,0,0,.2);border-radius:4px;font-size:12px}
  .legend i{width:16px;height:16px;float:left;margin-right:6px;opacity:.85}
  #title{font-weight:bold;padding:6px 10px}
</style></head><body>
<div id="title">%%TITLE%%</div>
<div id="controls">
  <button id="play-btn">⏸ Pause</button>
  <span id="day-label">Day 0</span>
  <input id="frame-slider" type="range" min="0" max="0" value="0"/>
  <label>Speed
    <select id="speed-select">
      <option value="250">Slow</option>
      <option value="120" selected>Normal</option>
      <option value="60">Fast</option>
    </select>
  </label>
</div>
<div id="map"></div>
<script>
const countiesGeojson = %%GEOJSON%%;
const frames = %%FRAMES%%;

function getColor(v){
  return v > 0.20 ? "#67000d" :
         v > 0.10 ? "#a50f15" :
         v > 0.05 ? "#cb181d" :
         v > 0.02 ? "#ef3b2c" :
         v > 0.01 ? "#fb6a4a" :
         v > 0.005 ? "#fc9272" :
         v > 0.001 ? "#fcbba1" :
         v > 0 ? "#fee0d2" :
                 "#f7f7f7";
}
const map = L.map("map");
function frameVal(fi, geoid){ const v = frames[fi].values[geoid]; return v ? v : 0; }
function styleFeature(feature){
  const geoid = String(feature.properties.GEOID);
  return {fillColor:getColor(frameVal(0, geoid)), weight:0.5, opacity:1,
          color:"#555", fillOpacity:0.85};
}
const countyLayer = L.geoJSON(countiesGeojson, {
  style: styleFeature,
  onEachFeature: (feature, layer) => {
    layer.on("mouseover", () => {
      const geoid = String(feature.properties.GEOID);
      const v = frameVal(currentFrame, geoid);
      layer.bindTooltip(
        "<b>"+feature.properties.NAME+"</b><br>GEOID "+geoid+
        "<br>I/N = "+v.toFixed(4), {sticky:true}).openTooltip();
    });
  }
}).addTo(map);
map.fitBounds(countyLayer.getBounds());

const layersByGeoid = {};
countyLayer.eachLayer(l => { layersByGeoid[String(l.feature.properties.GEOID)] = l; });

function updateFrame(fi){
  const frame = frames[fi];
  countyLayer.eachLayer(layer => {
    const geoid = String(layer.feature.properties.GEOID);
    layer.setStyle({fillColor:getColor(frameVal(fi, geoid))});
  });
  document.getElementById("day-label").innerText = "Day " + frame.day.toFixed(1);
  document.getElementById("frame-slider").value = fi;
}

let currentFrame = 0, playing = true, timer = null, intervalMs = 120;
function play(){
  if(timer !== null) return;
  playing = true; document.getElementById("play-btn").innerText = "⏸ Pause";
  timer = setInterval(() => {
    currentFrame = (currentFrame + 1) % frames.length;
    updateFrame(currentFrame);
  }, intervalMs);
}
function pause(){
  playing = false; document.getElementById("play-btn").innerText = "▶ Play";
  if(timer !== null){ clearInterval(timer); timer = null; }
}
document.getElementById("play-btn").addEventListener("click", () => {
  if(playing) pause(); else play();
});
document.getElementById("frame-slider").addEventListener("input", e => {
  pause(); currentFrame = Number(e.target.value); updateFrame(currentFrame);
});
document.getElementById("speed-select").addEventListener("change", e => {
  intervalMs = Number(e.target.value);
  if(playing){ pause(); play(); }
});

const legend = L.control({position:"bottomright"});
legend.onAdd = function(){
  const div = L.DomUtil.create("div","legend");
  const grades = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20];
  div.innerHTML = "<b>I / N</b><br>";
  for(let i=0;i<grades.length;i++){
    const from = grades[i];
    div.innerHTML += '<i style="background:'+getColor(from+0.0001)+'"></i> '+
      from + (grades[i+1] ? "&ndash;"+grades[i+1]+"<br>" : "+<br>");
  }
  return div;
};
legend.addTo(map);

document.getElementById("frame-slider").max = frames.length - 1;
updateFrame(0);
play();  // 로드 후 자동 재생
</script></body></html>
"""
    return (tmpl.replace("%%TITLE%%", title)
                .replace("%%GEOJSON%%", geojson_str)
                .replace("%%FRAMES%%", frames_str))


# ── 캐시: GeoJSON 로딩 + graph 생성 ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_graph(state_label):
    gdf = compute_centroids(load_counties(state=resolve_state(state_label)))
    G = build_adjacency_graph(gdf)
    return gdf, G


# ── 캐시: census 인구 (GEOID→인구) ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_population_map(state_label):
    gdf, _ = load_graph(state_label)
    pop, _ = load_county_population(gdf, verbose=False)
    return pop


# ── 캐시: 항공 operator (T-100 로딩 + catchment; D_air 은 나중에 주입) ──
@st.cache_data(show_spinner=False)
def load_flight_operator(state_label, flight_month, catchment_km):
    """
    무거운 부분(T-100 zip 읽기 + 공항 catchment 계산)만 캐시.
    D_air 은 값만 바꾸면 되므로 캐시 키에서 제외 → 슬라이더 바꿔도 재계산 안 함.
    """
    gdf, _ = load_graph(state_label)
    pops = load_population_map(state_label)
    y, m = int(flight_month[:4]), int(flight_month[4:])
    return build_flight_operator(
        gdf, y, m, radius_km=catchment_km, populations=pops,
        D_air=1.0, symmetric=True, verbose=False)


# ── 캐시: edge weight (CSV 캐시 재사용; geometry 재계산 안 함) ──
@st.cache_data(show_spinner=False)
def load_edge_weights(state_label):
    gdf, G = load_graph(state_label)
    G2 = G.copy()
    compute_edge_weights_border_distance(
        gdf, G2, cache_path=edge_weight_cache_path(resolve_state(state_label)))
    return {(a, b): d["weight"] for a, b, d in G2.edges(data=True)}


# ── 시뮬레이션 파이프라인 (버튼 눌렀을 때만 호출) ───────────────
def run_pipeline(p, progress, status):
    t0 = perf_counter()

    status.write("1/6 · Loading county graph…"); progress.progress(5)
    gdf, G = load_graph(p["state"])          # 캐시된 복사본
    G = G.copy()                              # 이번 실행용(안전하게 변형)

    if p["seed_geoid"] not in G.nodes:
        raise ValueError(f"seed GEOID '{p['seed_geoid']}' is not in the {p['state']} graph.")

    status.write("2/6 · Applying census population…"); progress.progress(20)
    populations = load_population_map(p["state"])

    status.write("3/6 · Initializing S/I/R…"); progress.progress(30)
    initialize_sir_compartments(
        G, default_population=10000,
        initial_infected_geoids=[p["seed_geoid"]], initial_I=p["initial_infected"],
        populations=populations)

    status.write("4/6 · Loading edge weights…"); progress.progress(45)
    if p["use_weighted"]:
        weights = load_edge_weights(p["state"])
        for (a, b), w in weights.items():
            if G.has_edge(a, b):
                G.edges[a, b]["weight"] = w
    # use_weighted=False 면 weight 속성 없음 → NumPy backend 가 w=1(unweighted)

    # 항공 layer (T-100). 무거운 부분은 캐시, D_air·rng 만 주입.
    flight = None
    if p["use_flight"]:
        status.write("4b/6 · Loading air travel data (T-100)…"); progress.progress(50)
        flight = dict(load_flight_operator(          # dict() 로 복사(캐시 원본 보호)
            p["state"], p["flight_month"], p["catchment_km"]))
        flight["D_air"] = p["D_air"]
        # 확률적 유입(기본): 감염자를 Poisson 정수로만 이동 → 0.001명 유령 감염 제거.
        # rng 는 캐시하지 않고 매 실행 새로 생성 → Run 할 때마다 다른 실현값(현실적 변동).
        flight["rng"] = np.random.default_rng()

    status.write("5/6 · Running NumPy simulation…"); progress.progress(55)
    gamma = 1.0 / p["recovery_days"]
    omega = (1.0 / p["immunity_days"]) if p["model"] == "sirs" else 0.0
    hist, geoids, S, I, R = run_reaction_diffusion_simulation_numpy(
        G, p["beta"], gamma, p["D_S"], p["D_I"], p["D_R"], p["dt"], p["days"],
        frame_nodes=list(gdf.index), frame_interval_days=p["frame_interval_days"],
        verbose=False, model=p["model"], omega=omega, flight=flight,
        ground="prevalence")   # 통근형: 인구 보존 (절대수 라플라시안의 인구 유출 버그 수정)
    numpy_state_to_graph(G, geoids, S, I, R)   # 지도 시각화용 최종상태 반영

    status.write("6/6 · Saving figures…"); progress.progress(80)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{p['model']}_{p['state']}_{ts}"
    paths = {
        "timeseries": os.path.join(STREAMLIT_DIR, f"{base}_timeseries.png"),
        "infected": os.path.join(STREAMLIT_DIR, f"{base}_infected_counties.png"),
        "final_map": os.path.join(STREAMLIT_DIR, f"{base}_final_map.png"),
        "population_map": os.path.join(STREAMLIT_DIR, f"{base}_population_map.png"),
        "history": os.path.join(STREAMLIT_DIR, f"{base}_history.csv"),
        "params": os.path.join(STREAMLIT_DIR, f"{base}_parameters.json"),
        "leaflet_html": os.path.join(
            STREAMLIT_DIR, f"leaflet_animation_{p['model']}_{p['state']}_{ts}.html"),
    }
    plot_timeseries(hist, paths["timeseries"])
    plot_infected_counties(hist, paths["infected"])
    plot_final_map(gdf, G, p["days"], paths["final_map"])
    plot_population_map(gdf, G, p["days"], paths["population_map"])

    # Leaflet 애니메이션용 프레임 데이터(브라우저에서 색상만 갱신 → 매 프레임 재plot 안 함)
    status.write("Building frame data…"); progress.progress(90)
    frames = build_frames_from_hist(hist, list(gdf.index))

    # history DataFrame (per-step 값만)
    keys = ["time", "S", "I", "R", "N", "infected", "max_ratio", "min_pop", "max_pop"]
    hist_df = pd.DataFrame({k: hist[k] for k in keys})
    hist_df.to_csv(paths["history"], index=False)

    runtime = perf_counter() - t0
    # 요약·안정성 지표
    I_series = hist["I"]
    pk = int(np.argmax(I_series))
    finite = all(np.isfinite(x).all() for x in (S, I, R))
    result = {
        "gamma": gamma, "omega": omega,
        "peak_day": hist["time"][pk], "peak_I": I_series[pk],
        "final_S": hist["S"][-1], "final_I": hist["I"][-1], "final_R": hist["R"][-1],
        "max_infected_county": int(max(hist["infected"])),
        "pop_cons_err": hist["max_cons_err"],
        "min_comp": hist["global_min_comp"],
        "clamp_count": int(hist.get("clamp_count", 0)),
        "finite": finite,
        "runtime": runtime,
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "air_on": flight is not None,
        "air_edges": (len(flight["P"]) if flight is not None else 0),
        "air_counties": (len(set(flight["cnty_idx"].tolist()))
                         if flight is not None else 0),
    }
    # 파라미터 JSON 저장
    params_out = dict(p); params_out.update(
        {"gamma": gamma, "omega": omega, "R0_local": p["beta"] / gamma})
    with open(paths["params"], "w") as f:
        json.dump(params_out, f, indent=2, default=str)

    progress.progress(100); status.write("Done ✅")
    return hist, hist_df, result, paths, params_out, gdf, frames


# ══════════════════════ UI ══════════════════════
st.set_page_config(page_title="County Graph Disease Simulation Explorer",
                   layout="wide")
st.title("County Graph Disease Simulation Explorer")
st.caption("Reaction-diffusion SIR/SIRS model on the US county adjacency graph. "
           "Adjust parameters in the sidebar and click **Run Simulation** to run. "
           "Infection spreads along edges (shared borders); in SIRS immunity wanes and reinfection occurs.")

# ── 사이드바 파라미터 폼 (버튼 전에는 실행 안 함) ──
with st.sidebar.form("simulation_parameters"):
    st.header("Parameters")
    state = st.selectbox("State", ["CA", "US"], index=0)
    seed_geoid = st.text_input("Seed county GEOID", value="06075")
    model = st.selectbox("Model", ["sirs", "sir"], index=0)

    st.subheader("Epidemiological parameters")
    beta = st.slider("Beta (infection rate)", 0.0, 1.0, 0.16, 0.01)
    recovery_days = st.slider("Recovery days (→ gamma=1/this)", 1, 60, 21, 1)
    immunity_days = st.slider("Immunity days (SIRS only, → omega=1/this)",
                              10, 3650, 180, 10)

    st.subheader("Ground movement (commuting)")
    D_ground = st.slider(
        "D_ground (commuting scale)", 0.0, 0.20, 0.05, 0.005, format="%.3f",
        help="C = D·w·√(N_a·N_b) sets the number of daily commuters. "
             "0.05 = about 38 million commuters/day between counties nationwide (actual ≈40 million/day)")
    D_S = D_I = D_R = D_ground   # 통근자는 S/I/R 을 함께 실어 나름 → 인구 보존

    st.subheader("Numerics & other")
    dt = st.slider("dt", 0.01, 0.20, 0.05, 0.01)
    days = st.slider("days", 10, 1000, 365, 5)
    initial_infected = st.number_input("Initial infected (seed county)",
                                       1, 1_000_000, 10)
    use_weighted_edges = st.checkbox("Use weighted edges", value=True)

    st.subheader("Air travel (T-100)")
    use_flight = st.checkbox("Enable air travel layer", value=True,
                             help="Adds long-range infection jumps between counties via air travel")
    flight_month = st.selectbox("Flight month", ["201910", "201906", "201902"],
                                index=0, help="Oct / Jun (peak season) / Feb (off season)")
    D_air = st.slider("D_air (1.0=actual traffic, 0=air travel blocked)", 0.0, 2.0, 1.0, 0.1)
    catchment_km = st.slider("Airport catchment radius (km)", 50.0, 200.0, 100.0, 10.0,
                             help="Airport catchment radius. 100km recommended (corrects hub distortion)")

    st.subheader("Interactive map (Leaflet)")
    show_leaflet = st.checkbox("Show interactive Leaflet animation", value=True)
    frame_interval_days = st.slider("Animation frame interval days",
                                    0.5, 10.0, 3.0, 0.5)
    simplify_tolerance = st.slider("Map simplify tolerance (deg)",
                                   0.0, 0.1, 0.03, 0.005, format="%.3f")

    submitted = st.form_submit_button("Run Simulation", type="primary",
                                      width="stretch")

params = {
    "state": state, "seed_geoid": seed_geoid.strip(), "model": model,
    "beta": beta, "recovery_days": recovery_days, "immunity_days": immunity_days,
    "D_S": D_S, "D_I": D_I, "D_R": D_R, "dt": dt, "days": days,
    "initial_infected": int(initial_infected),
    "frame_interval_days": frame_interval_days,
    "use_weighted": use_weighted_edges,
    "show_leaflet": show_leaflet, "simplify_tolerance": simplify_tolerance,
    "use_flight": use_flight, "flight_month": flight_month,
    "D_air": D_air, "catchment_km": catchment_km,
}

# ── 무거운 시뮬레이션 / 프레임 경고 (item 19) ──
n_steps = int(round(days / dt))
if days > 1000 or dt < 0.01:
    st.sidebar.error("Set days ≤ 1000 and dt ≥ 0.01.")
elif n_steps > 30000 or (state == "US" and n_steps > 15000):
    st.sidebar.warning("This simulation may be slow. Try increasing dt or reducing days.")
estimated_frames = days / frame_interval_days
if estimated_frames > 300:
    st.sidebar.warning("Too many animation frames. "
                       "Increase frame_interval_days for faster preview.")

if not submitted:
    st.info("Adjust parameters in the sidebar, then click Run Simulation.")
    st.stop()

# ── 실행 (버튼 눌렀을 때만) ──
progress = st.progress(0, text="Running…")
status = st.empty()
try:
    hist, hist_df, res, paths, params_out, gdf_sim, frames = run_pipeline(
        params, progress, status)
except Exception as e:
    progress.empty()
    st.error("Simulation failed.")
    st.exception(e)
    st.stop()

# ── Interactive Leaflet animation (최상단, 브라우저에서 색상만 갱신) ──
leaflet_html = None
if params["show_leaflet"]:
    st.subheader("Interactive Leaflet animation")
    try:
        gdf_prev = gdf_sim.to_crs("EPSG:4326").reset_index()   # GEOID→컬럼
        gdf_prev["geometry"] = gdf_prev.geometry.simplify(
            params["simplify_tolerance"], preserve_topology=True)
        title = f"{params['model'].upper()} · {params['state']} · seed {params['seed_geoid']}"
        leaflet_html = build_leaflet_animation_html(gdf_prev, frames, title=title)
        with open(paths["leaflet_html"], "w") as f:
            f.write(leaflet_html)
        st.caption(f"{len(frames)} frames · only county colors update (no re-plot per frame)")
        components.html(leaflet_html, height=720, scrolling=False)
    except Exception as e:
        # Leaflet 실패해도 나머지 결과는 계속 표시
        st.warning("Failed to build the interactive map. Static maps are shown below.")
        st.exception(e)

# ── 요약 metrics ──
st.subheader("Summary")
c = st.columns(4)
c[0].metric("Model", params["model"])
c[1].metric("State", params["state"])
c[2].metric("Seed county", params["seed_geoid"])
c[3].metric("Simulation runtime", f"{res['runtime']:.2f} s")
c = st.columns(4)
c[0].metric("Beta", f"{params['beta']:.3f}")
c[1].metric("Gamma", f"{res['gamma']:.4f}")
c[2].metric("Recovery days", params["recovery_days"])
c[3].metric("R0 (local)", f"{params['beta']/res['gamma']:.2f}")
c = st.columns(4)
c[0].metric("Omega", f"{res['omega']:.5f}")
c[1].metric("Immunity days", params["immunity_days"] if params["model"] == "sirs" else "—")
c[2].metric("Peak infected day", f"{res['peak_day']:.1f}")
c[3].metric("Peak total infected", f"{res['peak_I']:,.0f}")
c = st.columns(4)
c[0].metric("Final S", f"{res['final_S']:,.0f}")
c[1].metric("Final I", f"{res['final_I']:,.0f}")
c[2].metric("Final R", f"{res['final_R']:,.0f}")
c[3].metric("Max infected county", f"{res['max_infected_county']} / {res['n_nodes']}")
c = st.columns(4)
c[0].metric("D_ground (commuting)", f"{params['D_S']:.3f}")
c[1].metric("Pop conservation error", f"{res['pop_cons_err']:.2e}")
c[2].metric("Min compartment", f"{res['min_comp']:.2e}")
c[3].metric("Clamp count", res["clamp_count"])
c = st.columns(4)
c[0].metric("Air travel", "ON" if res["air_on"] else "OFF")
c[1].metric("D_air", f"{params['D_air']:.1f}" if res["air_on"] else "—")
c[2].metric("Flight month", params["flight_month"] if res["air_on"] else "—")
c[3].metric("Airport edges", f"{res['air_edges']:,}" if res["air_on"] else "—")
if res["air_on"]:
    st.caption(
        f"Air layer: {res['air_edges']:,} airport edges connect to {res['air_counties']} counties "
        f"({params['catchment_km']:.0f}km catchment radius). Passengers are symmetrized as P=(Pab+Pba)/2 to "
        f"conserve county population, and move based on prevalence (I/N). "
        f"**Stochastic importation**: only an **integer** number of infected drawn from a Poisson move, so "
        f"arrival times differ on every Run (realistic variability). "
        f"Run multiple times and look at the distribution before drawing conclusions."
        + ("" if params["state"] == "US"
           else "  ⚠️ With State=CA only routes within California are included (select US for inter-state routes)."))

# ── Time series ──
st.subheader("Time series (total S / I / R)")
st.image(paths["timeseries"], width="stretch")

st.subheader("Infected county count")
st.image(paths["infected"], width="stretch")

# ── 정적 지도 (기존 matplotlib PNG 유지) ──
col1, col2 = st.columns(2)
with col1:
    st.subheader("Final infection map (I / N)")
    st.image(paths["final_map"], width="stretch")
with col2:
    st.subheader("Population map (N)")
    st.image(paths["population_map"], width="stretch")

# ── history 테이블 ──
st.subheader("History")
st.dataframe(hist_df, width="stretch", height=260)

# ── 다운로드 버튼 ──
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "Download history CSV", data=hist_df.to_csv(index=False).encode("utf-8"),
    file_name=os.path.basename(paths["history"]), mime="text/csv")
d2.download_button(
    "Download parameters JSON",
    data=json.dumps(params_out, indent=2, default=str).encode("utf-8"),
    file_name=os.path.basename(paths["params"]), mime="application/json")
d3.download_button(
    "Download interactive HTML map",
    data=(leaflet_html or "").encode("utf-8"),
    file_name="leaflet_simulation_map.html", mime="text/html",
    disabled=(leaflet_html is None))

# ── 안정성 진단 (item 20) — 화면 맨 아래 ──
st.subheader("Diagnostics")
issues = []
if not res["finite"]:
    issues.append("NaN/Inf values occurred.")
if res["min_comp"] < -1e-6:
    issues.append(f"Negative compartment (min={res['min_comp']:.2e}). Reduce D or dt.")
if res["clamp_count"] > 0:
    issues.append(f"Clamped {res['clamp_count']} times (instability signal). Reducing D/dt recommended.")
if abs(res["pop_cons_err"]) > max(1.0, res["final_S"] + res["final_I"] + res["final_R"]) * 1e-6:
    issues.append(f"Large population conservation error ({res['pop_cons_err']:.2e}).")
if issues:
    for m in issues:
        st.warning(m)
else:
    st.success("Stability OK: no negatives or NaNs, population conserved.")

st.caption(f"Result files saved to: {STREAMLIT_DIR}")
