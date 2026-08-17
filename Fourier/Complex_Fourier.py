import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


name = "Complex3"    # 저장 파일명에 쓸 함수 이름
N_max = 50           # 마지막 항 개수

x = np.linspace(-np.pi, np.pi, 2000)
y = x**2
w = 1
c0 = np.pi**2 / 3


def partial_sum(N):
    s = np.zeros_like(x, dtype=complex)
    for n in range(-N, N+1):
        if n == 0:
            continue
        s = s + (2 * (-1)**n / n**2) * np.exp(1j*n*x)
    s = s + c0
    return s


fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(x, y, "k--", lw=2, label="f(x) = x^2 (target)")
line_re, = ax.plot([], [], "crimson",   lw=2, label="Fourier sum (real part)")
line_im, = ax.plot([], [], "royalblue", lw=2, label="imaginary part (~0)")

ax.set_xlim(-x.max(), x.max())
ax.set_ylim(-1, 10)         
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")


def update(N):
    s = partial_sum(N)
    line_re.set_data(x, s.real)
    line_im.set_data(x, s.imag)       
    ax.set_title(f"Complex Fourier series,  N = {N}")
    return line_re, line_im


ani = animation.FuncAnimation(fig, update, frames=range(1, N_max+1),
                              interval=200, blit=False)

fig.subplots_adjust(top=0.9)
ani.save(f"{name}_N{N_max}.gif", writer="pillow", fps=5)   
