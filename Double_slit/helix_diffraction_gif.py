"""
구를 1개 -> 40개까지 하나씩 추가하는 애니메이션을 GIF 로 저장.
(helix_diffraction.py 의 Enter 인터랙션을 그대로 프레임으로 굽는 버전)
"""

import matplotlib
matplotlib.use("Agg")           # 창 없이 렌더
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 파라미터 (helix_diffraction.py 와 동일, GIF용으로 조금 가볍게) ----------------
PER_HELIX = 20
R_HELIX = 1.0
DZ = 0.5
PITCH = 4.0
DPHI = 2 * np.pi * DZ / PITCH
STRAND_OFFSET = 0.375 * PITCH
SPHERE_R = 0.22

QMAX = 9.0
NPIX = 170
BEAM_STOP_Q = 1.0
N_ROT = 10
SIGMA_ATOM = 0.11

PAPER, SPOT_DARK = 0.82, 0.04
FOG_AMP, FOG_WIDTH, GRAIN = 0.10, 4.5, 0.015

COL1, COL2 = "#e8622a", "#2f8fd0"
zshift = (PER_HELIX - 1) * DZ / 2


# ---------------- 나선 & 회절 ----------------
def one_helix(n, z0):
    j = np.arange(n)
    phi = j * DPHI
    z = j * DZ + z0 - zshift
    return np.stack([R_HELIX * np.cos(phi), R_HELIX * np.sin(phi), z], axis=1)


def build_helices(N):
    n1, n2 = min(N, PER_HELIX), max(0, N - PER_HELIX)
    return one_helix(n1, 0.0), one_helix(n2, STRAND_OFFSET), n1, n2


def diffraction(atoms):
    qx = np.linspace(-QMAX, QMAX, NPIX)
    QX, QZ = np.meshgrid(qx, qx)
    Q = np.stack([QX, QZ], axis=-1)
    z = atoms[:, 2]
    Iacc = np.zeros((NPIX, NPIX))
    for k in range(N_ROT):
        a = 2 * np.pi * k / N_ROT
        xr = atoms[:, 0] * np.cos(a) - atoms[:, 1] * np.sin(a)
        R2 = np.stack([xr, z], axis=1)
        phase = np.tensordot(Q, R2.T, axes=([2], [0]))
        Iacc += np.abs(np.exp(1j * phase).sum(axis=-1)) ** 2
    I = Iacc / N_ROT
    I *= np.exp(-(SIGMA_ATOM ** 2) * (QX**2 + QZ**2))
    return QX, QZ, I / I.max()


# ---------------- 그림 준비 ----------------
_su, _sv = np.mgrid[0:2 * np.pi:12j, 0:np.pi:7j]
SPX, SPY, SPZ = np.cos(_su) * np.sin(_sv), np.sin(_su) * np.sin(_sv), np.cos(_sv)
NOISE = np.random.default_rng(0).normal(0.0, GRAIN, (NPIX, NPIX))  # 정적 필름 그레인

fig = plt.figure(figsize=(12, 6.0))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)


def render(n):
    h1, h2, N1, N2 = build_helices(n)
    atoms = np.vstack([h1, h2]) if N2 else h1

    ax1.cla()
    for nn, z0, col in [(N1, 0.0, COL1), (N2, STRAND_OFFSET, COL2)]:
        if nn >= 2:
            tt = np.linspace(0, (nn - 1) * DPHI, 200)
            zz = np.linspace(0, (nn - 1) * DZ, 200) + z0 - zshift
            ax1.plot(R_HELIX * np.cos(tt), R_HELIX * np.sin(tt), zz,
                     color=col, lw=1.4, alpha=0.55)
    for pts, col in [(h1, COL1), (h2, COL2)]:
        for (ax_, ay_, az_) in pts:
            ax1.plot_surface(ax_ + SPHERE_R * SPX, ay_ + SPHERE_R * SPY,
                             az_ + SPHERE_R * SPZ, color=col, shade=True,
                             linewidth=0, antialiased=False)
    ax1.quiver(0, -3.2, 0, 0, 1.4, 0, color="#3a80c9", lw=2, arrow_length_ratio=0.3)
    ax1.set_title(f"P atoms: {n}  (strand 1: {N1}" + (f" + strand 2: {N2}" if N2 else "") + ")",
                  fontsize=11)
    ax1.set_box_aspect((1, 1, 1.6))
    ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.view_init(elev=12, azim=-70)

    QX, QZ, I = diffraction(atoms)
    RR = np.sqrt(QX**2 + QZ**2)
    spots = (I + FOG_AMP * np.exp(-(RR / FOG_WIDTH) ** 2)) ** 0.45
    spots /= spots.max()
    film = PAPER - (PAPER - SPOT_DARK) * spots + NOISE
    film = np.clip(film, 0, 1)
    film = np.where(RR < BEAM_STOP_Q, 0.92, film)
    ax2.cla()
    ax2.imshow(film, extent=[-QMAX, QMAX, -QMAX, QMAX], origin="lower",
               cmap="gray", vmin=0, vmax=1)
    ax2.add_patch(plt.Circle((0, 0), BEAM_STOP_Q, facecolor="none",
                             edgecolor="0.45", lw=0.8, ls="--"))
    ax2.set_xlim(-QMAX, QMAX); ax2.set_ylim(-QMAX, QMAX)
    ax2.set_title("Diffraction (Photo 51 style)", fontsize=12)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_aspect("equal")
    fig.suptitle(f"Adding spheres one by one  —  {n}/40", fontsize=13)


def update(frame):
    n = frame + 1
    render(n)
    print(f"frame {n}/40", flush=True)
    return []


anim = FuncAnimation(fig, update, frames=40, interval=180)
anim.save("helix_diffraction.gif", writer=PillowWriter(fps=5), dpi=80)
print("저장 완료: helix_diffraction.gif")
