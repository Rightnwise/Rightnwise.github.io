"""
numpy_backend.py
================================================================
graph-based reaction-diffusion SIR 의 '빠른 NumPy 백엔드'.

모델·방정식·파라미터 의미는 기존 NetworkX 버전과 100% 동일하다.
바뀐 것은 '어떻게 계산하느냐'뿐이다:
  · 시뮬레이션 시작 전에 G 를 NumPy array(S,I,R + edge_u,edge_v,edge_w)로 변환
  · timestep loop 안에서는 G.nodes/G.edges 를 절대 접근하지 않고 array 만 사용
  · diffusion 은 edge 벡터 인덱싱 + np.add.at 으로 pairwise flux 누적
GeoPandas/NetworkX 는 graph 생성·weight 계산·시각화에만 쓰고
loop 안에서는 쓰지 않는다.

NetworkX 함수(step_reaction_diffusion_sir 등)는 그대로 두고, 이 파일은 '추가'다.
"""

import numpy as np
from time import perf_counter

from local_sir import INFECTED_THRESHOLD


# ── 1. G → NumPy state ───────────────────────────────────────────
def graph_to_numpy_state(G):
    """
    G 의 노드/엣지를 NumPy array 로 변환한다.
    반환: geoids, idx, S, I, R, edge_u, edge_v, edge_w
    edge_w 는 G.edges[a,b]['weight'] (없으면 1.0 = unweighted).
    """
    geoids = list(G.nodes())
    idx = {g: k for k, g in enumerate(geoids)}

    S = np.array([G.nodes[g]["S"] for g in geoids], dtype=float)
    I = np.array([G.nodes[g]["I"] for g in geoids], dtype=float)
    R = np.array([G.nodes[g]["R"] for g in geoids], dtype=float)

    eu, ev, ew = [], [], []
    for a, b, d in G.edges(data=True):
        eu.append(idx[a]); ev.append(idx[b]); ew.append(d.get("weight", 1.0))
    edge_u = np.array(eu, dtype=int)
    edge_v = np.array(ev, dtype=int)
    edge_w = np.array(ew, dtype=float)
    return geoids, idx, S, I, R, edge_u, edge_v, edge_w


# ── 2. NumPy state → G (최종 상태 반영) ──────────────────────────
def numpy_state_to_graph(G, geoids, S, I, R):
    """시뮬레이션 종료 후 최종 S/I/R/N 을 다시 G 노드에 기록(시각화용)."""
    for k, g in enumerate(geoids):
        d = G.nodes[g]
        d["S"] = float(S[k]); d["I"] = float(I[k]); d["R"] = float(R[k])
        d["N"] = float(S[k] + I[k] + R[k])
    return G


# ── 3. 한 timestep (벡터화) ──────────────────────────────────────
def step_reaction_diffusion_sir_numpy(S, I, R, edge_u, edge_v, edge_w,
                                      beta, gamma, D_S, D_I, D_R, dt):
    """
    NetworkX step 과 동일한 식의 벡터화 버전.
    ① local SIR reaction (N==0 division 안전) → ② weighted Laplacian diffusion.
    반환: (S, I, R, n_neg, min_val)  — n_neg: 음수 개수, min_val: clamp 전 최소값.
    """
    # ① reaction
    N = S + I + R
    safeN = np.where(N > 0, N, 1.0)
    local_infection = np.where(N > 0, beta * S * I / safeN, 0.0)
    new_infections = np.minimum(local_infection * dt, S)
    new_recoveries = np.minimum(gamma * I * dt, I)
    S = S - new_infections
    I = I + new_infections - new_recoveries
    R = R + new_recoveries

    # ② diffusion (pairwise flux, np.add.at 누적)
    flux_S = D_S * edge_w * (S[edge_u] - S[edge_v]) * dt
    flux_I = D_I * edge_w * (I[edge_u] - I[edge_v]) * dt
    flux_R = D_R * edge_w * (R[edge_u] - R[edge_v]) * dt

    delta_S = np.zeros_like(S); delta_I = np.zeros_like(I); delta_R = np.zeros_like(R)
    np.add.at(delta_S, edge_u, -flux_S); np.add.at(delta_S, edge_v, flux_S)
    np.add.at(delta_I, edge_u, -flux_I); np.add.at(delta_I, edge_v, flux_I)
    np.add.at(delta_R, edge_u, -flux_R); np.add.at(delta_R, edge_v, flux_R)

    S = S + delta_S; I = I + delta_I; R = R + delta_R

    # ④ 음수 안전장치: 조용히 무시하지 않고 개수/최소값 기록 후 필요시 clamp
    min_val = float(min(S.min(), I.min(), R.min()))
    n_neg = int((S < -1e-9).sum() + (I < -1e-9).sum() + (R < -1e-9).sum())
    if n_neg > 0:
        S = np.maximum(S, 0.0); I = np.maximum(I, 0.0); R = np.maximum(R, 0.0)
    return S, I, R, n_neg, min_val


# ── 3-b. SIRS step (면역 소실 ω·R 추가) ─────────────────────────
def step_reaction_diffusion_sirs_numpy(S, I, R, edge_u, edge_v, edge_w,
                                       beta, gamma, omega,
                                       D_S, D_I, D_R, dt):
    """
    SIR step 과 동일한 diffusion 구조에, local reaction 에만 면역소실 ω·R 추가.
    dS += ω R,  dR -= ω R  (recovered → susceptible).
    omega=0 이면 SIR step 과 완전히 동일.
    """
    # ① reaction (+ waning immunity)
    N = S + I + R
    safeN = np.where(N > 0, N, 1.0)
    infection_rate = np.where(N > 0, beta * S * I / safeN, 0.0)
    new_infections = np.minimum(infection_rate * dt, S)
    new_recoveries = np.minimum(gamma * I * dt, I)
    waning_immunity = np.minimum(omega * R * dt, R)
    S = S - new_infections + waning_immunity
    I = I + new_infections - new_recoveries
    R = R + new_recoveries - waning_immunity

    # ② diffusion (기존 pairwise flux 구조 그대로)
    flux_S = D_S * edge_w * (S[edge_u] - S[edge_v]) * dt
    flux_I = D_I * edge_w * (I[edge_u] - I[edge_v]) * dt
    flux_R = D_R * edge_w * (R[edge_u] - R[edge_v]) * dt
    delta_S = np.zeros_like(S); delta_I = np.zeros_like(I); delta_R = np.zeros_like(R)
    np.add.at(delta_S, edge_u, -flux_S); np.add.at(delta_S, edge_v, flux_S)
    np.add.at(delta_I, edge_u, -flux_I); np.add.at(delta_I, edge_v, flux_I)
    np.add.at(delta_R, edge_u, -flux_R); np.add.at(delta_R, edge_v, flux_R)
    S = S + delta_S; I = I + delta_I; R = R + delta_R

    min_val = float(min(S.min(), I.min(), R.min()))
    n_neg = int((S < -1e-9).sum() + (I < -1e-9).sum() + (R < -1e-9).sum())
    if n_neg > 0:
        S = np.maximum(S, 0.0); I = np.maximum(I, 0.0); R = np.maximum(R, 0.0)
    return S, I, R, n_neg, min_val


# ── 3-c. 유병률 기반 지상 확산 (통근형) ─────────────────────────
#
# 기존(절대수) : flux = D·w·(X_a − X_b)        → ΣX flux = D·w·(N_a−N_b) ≠ 0
#                → 사람이 큰 county 에서 작은 county 로 '영구 이주'(LA 970만→220만)
#
# 수정(유병률) : flux = C·(X_a/N_a − X_b/N_b)  → ΣX flux = C·(1−1) = 0
#                → 통근자는 왕복하므로 '감염'만 오가고 인구 N 은 정확히 보존
#                  C = 하루 통근자 수 = D · w · √(N_a·N_b)  (중력모형 근사)
#
# 주의: 세 compartment 가 같은 통근자에 실려 가므로 D 는 하나만 쓴다(D_S=D_I=D_R).
#       (감염자만 덜 이동시키려면 항공 layer 처럼 '좌석 보존' 보정이 필요)
def _prevalence_diffusion(S, I, R, edge_u, edge_v, C, dt):
    N = S + I + R
    safeN = np.where(N > 0, N, 1.0)
    dS = np.zeros_like(S); dI = np.zeros_like(I); dR = np.zeros_like(R)
    for X, dX in ((S, dS), (I, dI), (R, dR)):
        sX = np.where(N > 0, X / safeN, 0.0)
        flux = C * (sX[edge_u] - sX[edge_v]) * dt
        np.add.at(dX, edge_u, -flux)
        np.add.at(dX, edge_v, flux)
    return S + dS, I + dI, R + dR


def step_reaction_diffusion_sir_numpy_prevalence(S, I, R, edge_u, edge_v, C,
                                                 beta, gamma, dt):
    """SIR reaction + 유병률 기반 지상 확산(인구 보존)."""
    N = S + I + R
    safeN = np.where(N > 0, N, 1.0)
    new_inf = np.minimum(np.where(N > 0, beta * S * I / safeN, 0.0) * dt, S)
    new_rec = np.minimum(gamma * I * dt, I)
    S = S - new_inf
    I = I + new_inf - new_rec
    R = R + new_rec

    S, I, R = _prevalence_diffusion(S, I, R, edge_u, edge_v, C, dt)
    min_val = float(min(S.min(), I.min(), R.min()))
    n_neg = int((S < -1e-9).sum() + (I < -1e-9).sum() + (R < -1e-9).sum())
    if n_neg > 0:
        S = np.maximum(S, 0.0); I = np.maximum(I, 0.0); R = np.maximum(R, 0.0)
    return S, I, R, n_neg, min_val


def step_reaction_diffusion_sirs_numpy_prevalence(S, I, R, edge_u, edge_v, C,
                                                  beta, gamma, omega, dt):
    """SIRS reaction(+면역소실) + 유병률 기반 지상 확산(인구 보존)."""
    N = S + I + R
    safeN = np.where(N > 0, N, 1.0)
    new_inf = np.minimum(np.where(N > 0, beta * S * I / safeN, 0.0) * dt, S)
    new_rec = np.minimum(gamma * I * dt, I)
    waning = np.minimum(omega * R * dt, R)
    S = S - new_inf + waning
    I = I + new_inf - new_rec
    R = R + new_rec - waning

    S, I, R = _prevalence_diffusion(S, I, R, edge_u, edge_v, C, dt)
    min_val = float(min(S.min(), I.min(), R.min()))
    n_neg = int((S < -1e-9).sum() + (I < -1e-9).sum() + (R < -1e-9).sum())
    if n_neg > 0:
        S = np.maximum(S, 0.0); I = np.maximum(I, 0.0); R = np.maximum(R, 0.0)
    return S, I, R, n_neg, min_val


# ── 5. NumPy 시뮬레이션 ──────────────────────────────────────────
def run_reaction_diffusion_simulation_numpy(G, beta, gamma, D_S, D_I, D_R, dt,
                                            days, frame_nodes=None,
                                            frame_interval_days=1.0, verbose=True,
                                            model="sir", omega=0.0, flight=None,
                                            ground="absolute"):
    """
    NumPy 백엔드로 days 일 시뮬레이션. hist 는 NetworkX 버전과 같은 key 를 갖는다
    (+clamp_count). 애니메이션 frame 은 frame_interval_days 마다만 저장.
    반환: (hist, geoids, S, I, R)  — 최종 array 는 numpy_state_to_graph 로 반영.
    """
    geoids, idx, S, I, R, eu, ev, ew = graph_to_numpy_state(G)
    initial_total_N = float((S + I + R).sum())

    hist = {k: [] for k in ("time", "S", "I", "R", "N", "infected",
                            "max_ratio", "min_pop", "max_pop")}
    hist["frames"] = []
    hist["global_min_comp"] = float("inf")
    hist["max_cons_err"] = 0.0
    hist["clamp_count"] = 0

    n_steps = int(round(days / dt))
    frame_interval_steps = max(1, int(round(frame_interval_days / dt)))
    fn_pos = (np.array([idx[g] for g in frame_nodes], dtype=int)
              if frame_nodes is not None else None)

    def record(t, take_frame):
        N = S + I + R
        tS, tI, tR, tN = float(S.sum()), float(I.sum()), float(R.sum()), float(N.sum())
        ratios = np.where(N > 0, I / np.where(N > 0, N, 1.0), 0.0)
        hist["time"].append(t)
        hist["S"].append(tS); hist["I"].append(tI); hist["R"].append(tR); hist["N"].append(tN)
        hist["infected"].append(int((I > INFECTED_THRESHOLD).sum()))
        hist["max_ratio"].append(float(ratios.max()) if ratios.size else 0.0)
        hist["min_pop"].append(float(N.min())); hist["max_pop"].append(float(N.max()))
        hist["global_min_comp"] = min(hist["global_min_comp"],
                                      float(S.min()), float(I.min()), float(R.min()))
        hist["max_cons_err"] = max(hist["max_cons_err"],
                                   abs(tS + tI + tR - initial_total_N),
                                   abs(tN - initial_total_N))
        if fn_pos is not None and take_frame:
            hist["frames"].append({"t": t, "ratio": ratios[fn_pos].tolist(),
                                   "I": tI, "R": tR})

    # model·ground 에 따라 step 함수 선택 (기존 absolute 경로는 그대로 유지)
    if ground == "prevalence":
        # 통근자 수 C = D · w · √(N_a·N_b)  (중력모형 근사). N 은 보존되므로 상수.
        N0 = S + I + R
        C = D_S * ew * np.sqrt(N0[eu] * N0[ev])
        if verbose and not (D_S == D_I == D_R):
            print(f"[numpy] ⚠️ ground=prevalence 는 D 하나만 씁니다(D_S={D_S} 사용). "
                  f"세 compartment 가 같은 통근자에 실려야 인구가 보존됩니다.")
        if model == "sirs":
            def do_step(S, I, R):
                return step_reaction_diffusion_sirs_numpy_prevalence(
                    S, I, R, eu, ev, C, beta, gamma, omega, dt)
        else:
            def do_step(S, I, R):
                return step_reaction_diffusion_sir_numpy_prevalence(
                    S, I, R, eu, ev, C, beta, gamma, dt)
    elif model == "sirs":
        def do_step(S, I, R):
            return step_reaction_diffusion_sirs_numpy(
                S, I, R, eu, ev, ew, beta, gamma, omega, D_S, D_I, D_R, dt)
    else:
        def do_step(S, I, R):
            return step_reaction_diffusion_sir_numpy(
                S, I, R, eu, ev, ew, beta, gamma, D_S, D_I, D_R, dt)

    record(0.0, take_frame=True)
    if verbose:
        tag = f"model={model}" + (f", omega={omega}" if model == "sirs" else "")
        print(f"[numpy] {tag}, beta={beta}, gamma={gamma}, D_S={D_S}, D_I={D_I}, "
              f"D_R={D_R}, dt={dt}, days={days} ({n_steps} steps), edges={len(ew)}")

    # 항공 layer 사용 여부 (loop 밖에서 한 번만 준비)
    use_air = flight is not None and flight.get("D_air", 0.0) != 0.0
    if use_air:
        from flight_coupling import apply_flight_flux
        if verbose:
            print(f"[numpy] 항공 layer 사용: 공항 edge {len(flight['P']):,}개, "
                  f"D_air={flight['D_air']}")

    neg_warned = False
    for k in range(1, n_steps + 1):
        S, I, R, n_neg, min_val = do_step(S, I, R)
        # 항공 layer (연산자 분리): 유병률 기반 이동 → county 인구 N 은 정확히 보존
        if use_air:
            S, I, R = apply_flight_flux(S, I, R, S + I + R, flight, dt)
        if n_neg > 0:
            hist["clamp_count"] += n_neg
            if not neg_warned:
                neg_warned = True
                print(f"[numpy] ⚠️  step {k}: 음수 {n_neg}개(min={min_val:.3e}) 발생 "
                      f"→ D 또는 dt 를 줄이세요(현재 dt={dt}). 0 클램프 후 계속.")
        record(k * dt, take_frame=(k % frame_interval_steps == 0 or k == n_steps))

    if verbose:
        print(f"[numpy] 완료. population 보존 최대 오차 = {hist['max_cons_err']:.3e}")
        print(f"[numpy] 전 기간 최소 compartment = {hist['global_min_comp']:.3e}, "
              f"clamp 누적 = {hist['clamp_count']}")
        if frame_nodes is not None:
            print(f"[numpy] 애니메이션 프레임 {len(hist['frames'])}장 "
                  f"(간격 {frame_interval_days}일)")
    return hist, geoids, S, I, R


# ── 7. 백엔드 비교 검증 ──────────────────────────────────────────
def compare_networkx_vs_numpy_backend(G, beta, gamma, D_S, D_I, D_R, dt, days,
                                      networkx_step_fn=None, tol=1e-9,
                                      model="sir", omega=0.0):
    """
    동일 초기조건·파라미터로 NumPy 와 NetworkX 백엔드를 돌려 최종 상태를 비교.
    NumPy 를 먼저(=G 를 안 바꿈) 돌리고, 그 다음 NetworkX(=G 를 최종으로 변경).
    networkx_step_fn: weighted/sirs 비교 시 해당 step; None 이면 unweighted SIR.
    반환: dict(passed, 각 diff, t_np, t_nx, speedup)
    """
    from reaction_diffusion_sir import run_reaction_diffusion_sir

    # NumPy (G 를 수정하지 않음)
    t0 = perf_counter()
    _, geoids, Sn, In, Rn = run_reaction_diffusion_simulation_numpy(
        G, beta, gamma, D_S, D_I, D_R, dt, days, verbose=False,
        model=model, omega=omega)
    t_np = perf_counter() - t0

    # NetworkX (G 를 최종상태로 변경)
    t0 = perf_counter()
    run_reaction_diffusion_sir(G, beta, gamma, D_S, D_I, D_R, dt, days,
                               verbose=False, step_fn=networkx_step_fn)
    t_nx = perf_counter() - t0

    Sx = np.array([G.nodes[g]["S"] for g in geoids])
    Ix = np.array([G.nodes[g]["I"] for g in geoids])
    Rx = np.array([G.nodes[g]["R"] for g in geoids])

    dS = float(np.abs(Sn - Sx).max())
    dI = float(np.abs(In - Ix).max())
    dR = float(np.abs(Rn - Rx).max())
    total_N_diff = float(abs((Sn + In + Rn).sum() - (Sx + Ix + Rx).sum()))
    denom = max(1.0, float(np.abs(Sx).max()), float(np.abs(Ix).max()), float(np.abs(Rx).max()))
    passed = max(dS, dI, dR) / denom < tol
    speedup = (t_nx / t_np) if t_np > 0 else float("inf")

    print("\nBackend comparison:")
    print(f"max_abs_diff_S = {dS:.3e}")
    print(f"max_abs_diff_I = {dI:.3e}")
    print(f"max_abs_diff_R = {dR:.3e}")
    print(f"total_N_diff   = {total_N_diff:.3e}")
    print(f"NetworkX simulation time: {t_nx:.3f} s")
    print(f"NumPy    simulation time: {t_np:.3f} s")
    print(f"Speedup: {speedup:.1f}x")
    if passed:
        print("PASS: NumPy backend matches NetworkX backend within tolerance.\n")
    else:
        print(f"FAIL: 차이가 tolerance({tol}) 초과 — 모델 불일치 의심.\n")
    return {"passed": passed, "dS": dS, "dI": dI, "dR": dR,
            "total_N_diff": total_N_diff, "t_np": t_np, "t_nx": t_nx,
            "speedup": speedup}
