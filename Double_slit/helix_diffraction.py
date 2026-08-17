"""
================================================================
DNA 이중나선 회절 — 실제 Photo 51 처럼
================================================================

* 나선 축 = z(세로), 빛(평면파)은 y 방향 진행, 스크린은 뒤쪽 y=D 평면.
* 나선 위에 인 원자(구, 점 산란체)를 하나씩 붙인다. (개수는 아래 N_SPHERES)
* 각 원자가 빛을 같은 위상으로 산란 -> 먼 거리(Fraunhofer) 회절.
    F(q)=Σ exp(i q·r_j),  밝기 I=|F|²,  q=k_out-k_in

▼ 실제 DNA / Photo 51 처럼 보이게 한 두 가지 ▼
 (1) 두 가닥을 '같은 방향'으로 꼬되, 축을 따라 pitch의 3/8 만큼 어긋나게 배치.
     -> 이 어긋남 때문에 4번째 층선(layer line)이 사라지는 그 유명한 X 무늬가 나온다.
 (2) 섬유 회절(fiber diffraction): 실제 사진은 나선이 축 주위로 마구 돌아간
     것들의 평균이므로, 회전 평균을 해줘야 좌우 대칭의 깨끗한 X 가 된다.

----------------------------------------------------------------
▼▼▼ 구(인 원자) 개수 : 최소 1, 최대 40 ▼▼▼
   - 1~20  : 나선 1개
   - 21~40 : 첫 나선 20개 채운 뒤, 두 번째 나선을 만들어 아래에서부터 붙임 -> 이중나선
   (Photo 51 다운 X 무늬는 두 가닥이 다 있는 40개에서 가장 잘 보임)
"""
N_SPHERES = 20
# ▲▲▲ ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

PER_HELIX = 20
N_SPHERES = int(np.clip(N_SPHERES, 1, 2 * PER_HELIX))

# ---------------- 나선 & 빛 파라미터 ----------------
R_HELIX = 1.0
DZ = 0.5                         # 구 하나당 축 방향 상승량
PITCH = 4.0                      # 나선 한 바퀴 상승량 (주기). 클수록 X 팔이 벌어짐
DPHI = 2 * np.pi * DZ / PITCH
STRAND_OFFSET = 0.375 * PITCH    # 두 가닥의 축 방향 어긋남 = 3/8 pitch (핵심!)
SPHERE_R = 0.22

QMAX = 9.0                       # 역공간(q) 표시 범위 (작게 -> 안쪽 X 로 줌인)
NPIX = 200                       # (인터랙티브 응답성 위해 조금 낮춤)
BEAM_STOP_Q = 1.0                # 빔 스톱 (중앙 직진 빔 가림, q 단위)
N_ROT = 12                       # 섬유 회절: 축 둘레 회전 평균 횟수
SIGMA_ATOM = 0.11                # 원자 크기(형태인자): 클수록 바깥 번짐이 더 눌림
SHOW_XGUIDE = False              # X자 팔을 안내선으로 표시할지
XARM_SLOPE = 0.97                # 안내선 기울기 (q_x / q_z)

# 실제 Photo 51 = 필름 음화: 밝은 회색 배경 + 검은 스팟
PAPER = 0.82                     # 배경(필름) 밝기 - 밝은 회색
SPOT_DARK = 0.01               # 스팟 최대 어둠 (0=완전 검정)
FOG_AMP = 0.10                   # 중앙 확산 산란 헤일로 세기
FOG_WIDTH = 4.5                  # 헤일로 폭
GRAIN = 0.015                    # 필름 그레인(잡티) 세기


# ---------------- 1. 이중나선 원자 위치 ----------------
def one_helix(n, z0, zshift):
    """구 n개짜리 나선. 두 가닥 모두 같은 방향 꼬임(+), z0 만큼 축 방향 어긋남."""
    j = np.arange(n)
    phi = j * DPHI
    z = j * DZ + z0 - zshift
    return np.stack([R_HELIX * np.cos(phi), R_HELIX * np.sin(phi), z], axis=1)


def build_helices(N):
    n1 = min(N, PER_HELIX)
    n2 = max(0, N - PER_HELIX)
    zshift = (PER_HELIX - 1) * DZ / 2
    h1 = one_helix(n1, 0.0, zshift)              # 첫 가닥
    h2 = one_helix(n2, STRAND_OFFSET, zshift)    # 둘째 가닥: 같은 꼬임 + 3/8 pitch 어긋남
    return h1, h2, n1, n2


# ---------------- 2. 섬유 회절 (평평한 역공간 qx-qz, 회전 평균) ----------------
def diffraction(atoms):
    # 검출면을 역공간 (qx, qz) 평면으로 (qy=0 투영) -> 층선이 곧게 서서 X자가 또렷.
    qx = np.linspace(-QMAX, QMAX, NPIX)
    qz = np.linspace(-QMAX, QMAX, NPIX)
    QX, QZ = np.meshgrid(qx, qz)
    Q = np.stack([QX, QZ], axis=-1)                 # (NPIX,NPIX,2)
    z = atoms[:, 2]

    Iacc = np.zeros((NPIX, NPIX))
    for k in range(N_ROT):                          # 나선을 축(z) 둘레로 돌려가며 평균
        a = 2 * np.pi * k / N_ROT
        xr = atoms[:, 0] * np.cos(a) - atoms[:, 1] * np.sin(a)  # qx 축 방향 성분
        R2 = np.stack([xr, z], axis=1)              # (natoms, 2)
        phase = np.tensordot(Q, R2.T, axes=([2], [0]))          # (NPIX,NPIX,natoms)
        Iacc += np.abs(np.exp(1j * phase).sum(axis=-1)) ** 2
    I = Iacc / N_ROT
    # 원자 형태인자(가우시안): 바깥쪽(높은 q) 번짐을 눌러 안쪽 X를 또렷하게
    I *= np.exp(-(SIGMA_ATOM ** 2) * (QX**2 + QZ**2))
    return QX, QZ, I / I.max()


# ---------------- 3. 인터랙티브 그리기 (Enter = 구 추가, Backspace = 제거) ----------------
COL1, COL2 = "#e8622a", "#2f8fd0"
zshift = (PER_HELIX - 1) * DZ / 2

# 구 표면 메쉬 (한 번만 계산)
_su, _sv = np.mgrid[0:2 * np.pi:12j, 0:np.pi:7j]
SPX, SPY, SPZ = np.cos(_su) * np.sin(_sv), np.sin(_su) * np.sin(_sv), np.cos(_sv)
_rng = np.random.default_rng(0)

fig = plt.figure(figsize=(13, 6.4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)

state = {"n": 1}          # 현재 구(인 원자) 개수


def render():
    n = state["n"]
    h1, h2, N1, N2 = build_helices(n)
    atoms = np.vstack([h1, h2]) if N2 else h1

    # --- (좌) 3D 나선 ---
    ax1.cla()
    for nn, z0, col in [(N1, 0.0, COL1), (N2, STRAND_OFFSET, COL2)]:
        if nn >= 2:
            tt = np.linspace(0, (nn - 1) * DPHI, 300)
            zz = np.linspace(0, (nn - 1) * DZ, 300) + z0 - zshift
            ax1.plot(R_HELIX * np.cos(tt), R_HELIX * np.sin(tt), zz,
                     color=col, lw=1.4, alpha=0.55)
    for pts, col in [(h1, COL1), (h2, COL2)]:
        for (ax_, ay_, az_) in pts:
            ax1.plot_surface(ax_ + SPHERE_R * SPX, ay_ + SPHERE_R * SPY,
                             az_ + SPHERE_R * SPZ, color=col, shade=True,
                             linewidth=0, antialiased=True)
    ax1.quiver(0, -3.2, 0, 0, 1.4, 0, color="#3a80c9", lw=2, arrow_length_ratio=0.3)
    ax1.text(0, -3.5, 0, "빛(X선)", color="#2a6bb0", fontsize=10)
    strand_txt = f"가닥1: {N1}" + (f" + 가닥2: {N2}" if N2 else "")
    ax1.set_title(f"인(P) {n}개  ({strand_txt})", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z (나선 축)")
    ax1.set_box_aspect((1, 1, 1.6))
    ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
    ax1.view_init(elev=12, azim=-70)

    # --- (우) 회절 무늬 (Photo 51 음화) ---
    QX, QZ, I = diffraction(atoms)
    RR = np.sqrt(QX**2 + QZ**2)
    halo = FOG_AMP * np.exp(-(RR / FOG_WIDTH) ** 2)
    spots = (I + halo) ** 0.45
    spots /= spots.max()
    film = PAPER - (PAPER - SPOT_DARK) * spots
    film = film + _rng.normal(0.0, GRAIN, film.shape)
    film = np.clip(film, 0, 1)
    film = np.where(RR < BEAM_STOP_Q, 0.92, film)
    ax2.cla()
    ax2.imshow(film, extent=[-QMAX, QMAX, -QMAX, QMAX], origin="lower",
               cmap="gray", vmin=0, vmax=1)
    if BEAM_STOP_Q > 0:
        ax2.add_patch(plt.Circle((0, 0), BEAM_STOP_Q, facecolor="none",
                                 edgecolor="0.45", lw=0.8, ls="--"))
    ax2.set_xlim(-QMAX, QMAX); ax2.set_ylim(-QMAX, QMAX)
    ax2.set_facecolor("black")
    ax2.set_title("회절 무늬 (Photo 51 처럼)", fontsize=12)
    ax2.set_xlabel("q_x  (가로)")
    ax2.set_ylabel("q_z  (세로, 나선 축 방향)")
    ax2.set_aspect("equal")

    n_helix = 2 if N2 else 1
    fig.suptitle(
        f"[ Enter=구 추가 · Backspace=제거 ]   "
        f"현재 {n}개 (나선 {n_helix}개) / 최대 {2 * PER_HELIX}개",
        fontsize=13,
    )
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "enter" and state["n"] < 2 * PER_HELIX:
        state["n"] += 1
        render()
    elif event.key == "backspace" and state["n"] > 1:
        state["n"] -= 1
        render()


fig.canvas.mpl_connect("key_press_event", on_key)
render()
plt.tight_layout()
plt.show()
