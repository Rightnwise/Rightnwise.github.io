"""
Why a sphere behaves like a finite-width source.

A diffraction envelope is the Fourier transform of the object's shape:
  - finite slit (width w) -> sinc^2(pi w u / lambda)
  - sphere (diameter D)   -> sphere form factor   (same principle, different shape)
A finite size => an envelope. The sphere's diameter D plays the role of the slit width w.

Bottom panel: why we observe the SIDES, not the center.
  - Light that misses the sphere stays straight -> forms the bright DIRECT beam at the
    center (unscattered) -> blocked by a beam stop.
  - Light that hits the sphere is scattered to the SIDES -> this carries the shape info,
    so we observe there.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LAM = 0.5
u = np.linspace(-0.6, 0.6, 2000)
W = 2.0            # slit width
D = 2.0            # sphere diameter (= W for comparison)

COL_SLIT = "#1f7a4d"
COL_SPH = "#e8622a"


def env_slit(w):
    return np.sinc(w * u / LAM) ** 2                     # sinc(x)=sin(pi x)/(pi x)


def env_sphere(dia):
    R = dia / 2
    x = (2 * np.pi * u / LAM) * R
    f = np.where(np.abs(x) < 1e-6, 1.0,
                 3 * (np.sin(x) - x * np.cos(x)) / x**3)  # uniform-sphere form factor
    return f**2


fig = plt.figure(figsize=(12, 9.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], width_ratios=[1, 2.4],
                      hspace=0.45, wspace=0.15)

# ---------- Row 1 : finite slit ----------
axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])
axL.add_patch(Rectangle((-0.15, -W/2), 0.3, W, facecolor=COL_SLIT, edgecolor="none"))
axL.annotate("", xy=(0.55, -W/2), xytext=(0.55, W/2),
             arrowprops=dict(arrowstyle="<->", color=COL_SLIT, lw=1.8))
axL.text(0.72, 0, "w", color=COL_SLIT, fontsize=15, va="center")
axL.set_title("finite slit (width $w$)", fontsize=12)
axR.plot(u, env_slit(W), color=COL_SLIT, lw=2)
axR.fill_between(u, 0, env_slit(W), color=COL_SLIT, alpha=0.15)
axR.set_title(r"envelope: $\mathrm{sinc}^2(\pi w u/\lambda)$", fontsize=12)

# ---------- Row 2 : sphere ----------
axL = fig.add_subplot(gs[1, 0]); axR = fig.add_subplot(gs[1, 1])
axL.add_patch(Circle((0, 0), D/2, facecolor=COL_SPH, edgecolor="0.2", lw=1))
axL.annotate("", xy=(0.55, -D/2), xytext=(0.55, D/2),
             arrowprops=dict(arrowstyle="<->", color="#c25a1a", lw=1.8))
axL.text(0.72, 0, "D", color="#c25a1a", fontsize=15, va="center")
axL.set_title("sphere (diameter $D$)", fontsize=12)
axR.plot(u, env_sphere(D), color=COL_SPH, lw=2, label="sphere form factor")
axR.fill_between(u, 0, env_sphere(D), color=COL_SPH, alpha=0.15)
axR.plot(u, env_slit(W), "--", color=COL_SLIT, lw=1.4, alpha=0.8, label="slit $\\mathrm{sinc}^2$ (compare)")
axR.legend(fontsize=9, loc="upper right")
axR.set_title("envelope: sphere form factor  (same principle)", fontsize=12)

for axL in [fig.axes[0], fig.axes[2]]:
    axL.set_xlim(-1.3, 1.6); axL.set_ylim(-1.6, 1.6)
    axL.set_aspect("equal"); axL.axis("off")
for axR in [fig.axes[1], fig.axes[3]]:
    axR.set_ylim(0, 1.08); axR.set_xlim(u.min(), u.max())
    axR.grid(True, alpha=0.25); axR.set_ylabel("envelope")
fig.axes[3].set_xlabel(r"$u=\sin\theta$")

# ---------- Row 3 : scattering geometry (why observe the sides) ----------
ax = fig.add_subplot(gs[2, :])
xs = 5.4                      # screen position

# incoming plane wave
for xw in [-4.3, -3.9, -3.5]:
    ax.plot([xw, xw], [-2.6, 2.6], color="#4a90d9", lw=1.6, alpha=0.55)
for yw in [-2, -1, 0, 1, 2]:
    ax.annotate("", xy=(-1.1, yw), xytext=(-3.0, yw),
                arrowprops=dict(arrowstyle="->", color="#3a80c9", lw=1.3, alpha=0.85))

# sphere
ax.add_patch(Circle((0, 0), 0.62, facecolor=COL_SPH, edgecolor="0.2", lw=1))

# direct beam (missed the sphere) -> center of screen -> blocked
ax.annotate("", xy=(xs - 0.15, 0), xytext=(0.7, 0),
            arrowprops=dict(arrowstyle="->", color="0.45", lw=2.2))

# scattered light (hit the sphere) -> sides -> observe
for ys in [2.3, 1.5, -1.5, -2.3]:
    ax.annotate("", xy=(xs - 0.15, ys), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color=COL_SPH, lw=1.5, alpha=0.9))

# screen + beam stop + bright side spots
ax.plot([xs, xs], [-3, 3], color="black", lw=4)
ax.add_patch(Circle((xs, 0), 0.42, facecolor="black", edgecolor="0.4", lw=1))
for ys in [2.3, 1.5, -1.5, -2.3]:
    ax.add_patch(Ellipse((xs, ys), 0.16, 0.55, facecolor=COL_SPH, edgecolor="none", alpha=0.85))

ax.set_xlim(-4.8, 7.2); ax.set_ylim(-3.4, 3.4)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Why observe the sides, not the center", fontsize=12)

fig.suptitle("Why a sphere behaves like a finite-width source", fontsize=14)
plt.savefig("sphere_as_width.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
plt.show()
print("saved: sphere_as_width.png")
