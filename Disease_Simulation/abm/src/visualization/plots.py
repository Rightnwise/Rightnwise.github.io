"""
시각화 (모든 그래프 저장은 여기서만).
  단일 실행: epidemic_curve / cumulative_deaths / final_map
  비교:      scenario_runner 용 6종 비교 그래프
  스윕:      parameter_sweep 용 라인/히트맵
"""
import matplotlib
matplotlib.use("Agg")                       # 파일 저장 전용(GUI 불필요)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

STATE_COLOR = {"S": "#2f6fd0", "L": "#e58a1a", "I": "#d1352c",
               "R": "#2e8b57", "D": "#000000"}


def _mark_seed(ax, gdf):
    """최초 발생 동네(is_seed)에 별표 표시."""
    if "is_seed" not in gdf.columns or not gdf["is_seed"].any():
        return
    seed = gdf[gdf["is_seed"]].geometry.iloc[0].centroid
    ax.scatter([seed.x], [seed.y], marker="*", s=360, c="#1f6feb",
               edgecolors="white", linewidths=1.0, zorder=6)
    ax.annotate("Initial outbreak", (seed.x, seed.y), textcoords="offset points",
                xytext=(8, 8), fontsize=10, color="#1f6feb", fontweight="bold")


# ---------------- 단일 실행 ----------------
def plot_epidemic_curve(result, path):
    ts = result.timeseries
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts["day"], ts["S"], color=STATE_COLOR["S"], lw=2, label="S Susceptible")
    ax.plot(ts["day"], ts["latent_infected"], color=STATE_COLOR["L"], lw=2, label="L Latent (asymptomatic)")
    ax.plot(ts["day"], ts["I"], color=STATE_COLOR["I"], lw=2, label="I Symptomatic infected")
    ax.plot(ts["day"], ts["R"], color=STATE_COLOR["R"], lw=2, label="R Recovered")
    ax.plot(ts["day"], ts["D"], color=STATE_COLOR["D"], lw=2, label="D Deaths (cumulative)")
    ax.fill_between(ts["day"], 0, ts["active_infected"], color=STATE_COLOR["I"], alpha=0.10)
    ax.set_title(f"S/L/I/R/D over time — {result.summary['scenario_name']}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of agents")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cumulative_deaths(result, path):
    ts = result.timeseries
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts["day"], ts["cumulative_deaths"], color="#000000", lw=2.2, label="Cumulative deaths")
    ax.bar(ts["day"], ts["new_deaths"], color="#d1352c", alpha=0.4, label="Daily new deaths")
    ax.set_title(f"Cumulative / new deaths — {result.summary['scenario_name']}  "
                 f"(total {result.summary['total_deaths']:,})")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of deaths")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_final_map(result, path):
    gdf = result.final_node_states
    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, column="attack_rate", cmap="Reds", vmin=0, vmax=1,
             edgecolor="0.7", linewidth=0.3, legend=True,
             legend_kwds={"label": "Neighborhood final attack rate = ever-infected / residents",
                          "fraction": 0.035, "pad": 0.02})
    _mark_seed(ax, gdf)
    ax.set_title(f"Final attack rate map — {result.summary['scenario_name']}  "
                 f"(overall attack rate {result.summary['final_attack_rate']*100:.0f}%)")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------- 감염 확산 애니메이션(GIF) ----------------
def build_animation(result, path, fps=8):
    """동네별 감염률 시공간 확산을 GIF 로 렌더링.

    run_simulation(..., capture_spatial=True) 로 얻은 spatial_frames 필요."""
    if not result.spatial_frames:
        raise ValueError("spatial_frames 가 없습니다. "
                         "run_simulation(..., capture_spatial=True) 로 실행하세요.")
    gdf = result.final_node_states                 # geometry(노드 순서와 동일)
    frames = result.spatial_frames
    days = result.frame_days
    ts = result.timeseries.set_index("day")

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(ax=ax, color="#f4f4f4", edgecolor="0.8", linewidth=0.3)
    norm = Normalize(0, 1)
    gdf.plot(ax=ax, column=frames[0], cmap="Reds", norm=norm,
             edgecolor="0.7", linewidth=0.3)
    art = ax.collections[-1]
    _mark_seed(ax, gdf)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(ScalarMappable(norm=norm, cmap="Reds"), ax=ax,
                 fraction=0.035, pad=0.02).set_label("Neighborhood infection rate I/(alive N)")
    title = ax.set_title("")

    def update(i):
        art.set_array(frames[i])
        d = days[i]
        row = ts.loc[d] if d in ts.index else None
        if row is not None:
            title.set_text(f"{result.summary['scenario_name']}  "
                           f"Day {d}   I={int(row['active_infected']):,}  "
                           f"D={int(row['D']):,}")
        return art, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# ---------------- 시나리오 비교(6종) ----------------
def _line_compare(results, col, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, res in results.items():
        ts = res.timeseries
        ax.plot(ts["day"], ts[col], lw=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bar_compare(results, metric, title, ylabel, path):
    names = list(results.keys())
    vals = [results[n].summary[metric] for n in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(names, vals, color="#2f6fd0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=40)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.2f}" if isinstance(v, float) else f"{v:,}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scenario_comparison(results, outdir):
    """results: {name: SimulationResult}. 6종 비교 그래프를 outdir 에 저장."""
    import os
    _line_compare(results, "active_infected", "Active infected by scenario",
                  "Concurrent infected", os.path.join(outdir, "active_infected.png"))
    _line_compare(results, "cumulative_deaths", "Cumulative deaths by scenario",
                  "Cumulative deaths", os.path.join(outdir, "cumulative_deaths.png"))
    _line_compare(results, "new_infections", "Daily new infections by scenario",
                  "New infections / day", os.path.join(outdir, "new_infections.png"))
    _bar_compare(results, "peak_infected", "Peak infected by scenario",
                 "Peak concurrent infected", os.path.join(outdir, "peak_infected_bar.png"))
    _bar_compare(results, "total_deaths", "Total deaths by scenario",
                 "Total deaths", os.path.join(outdir, "total_deaths_bar.png"))
    _bar_compare(results, "final_attack_rate", "Final attack rate by scenario",
                 "Final attack rate", os.path.join(outdir, "final_attack_rate_bar.png"))


# ---------------- 파라미터 스윕 ----------------
def plot_sweep_line(df, x, y, path, title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    d = df.sort_values(x)
    ax.plot(d[x], d[y], "o-", color="#d1352c", lw=2)
    ax.set_title(title or f"{x} vs {y}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_heatmap(df, xcol, ycol, vcol, path, title=None):
    xs = sorted(df[xcol].unique())
    ys = sorted(df[ycol].unique())
    grid = np.full((len(ys), len(xs)), np.nan)
    for _, r in df.iterrows():
        grid[ys.index(r[ycol]), xs.index(r[xcol])] = r[vcol]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="magma", origin="lower", aspect="auto")
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
    ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title or f"{vcol}")
    for j in range(len(ys)):
        for i in range(len(xs)):
            if not np.isnan(grid[j, i]):
                ax.text(i, j, f"{grid[j, i]:,.0f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(vcol)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
