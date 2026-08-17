import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


name = "sawtooth"    # 저장 파일명에 쓸 함수 이름
N_max = 50           # 마지막 항 개수

# f(x) = 0 if -5 < x < 0, 3 if 0 < x < 5
# x = np.linspace(-15, 15, 3000)
# y = np.where(np.mod(x, 10) < 5, 3, 0)
x = np.linspace(-3*np.pi, 3*np.pi, 3000)
y = np.where((np.mod(x+np.pi, 2*np.pi) - np.pi) < 0, -1, 1)
# x = np.linspace(-12, 12, 3000)
# t = np.mod(x + 4, 8) - 4
# y = np.where(t < 0, -t - 4, -t + 4)


def partial_sum(N):
    """N까지의 푸리에 부분합."""
    s = np.zeros_like(x)
    for k in range( 1, N + 1):
        n = 2*k - 1
        # s = s + (6/np.pi) * (1/n) * np.sin(n*np.pi*x / 5)
        # s = s + (8 / np.pi) * (1/n) * np.sin(n*np.pi*x / 4)
        s = s + (4 / np.pi) * (1/n) * np.sin(n*x)
    return s


fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(x, y, color="black", lw=4, label="f(x) (target)")
line, = ax.plot([], [], color="crimson", lw=2, label="Fourier sum")

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-2, 2)
ax.axhline(0, color="0.7", lw=0.8)
ax.axvline(0, color="0.7", lw=0.8)
ax.grid(True, alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(loc="upper left")


def update(N):
    line.set_data(x, partial_sum(N))
    ax.set_title(f"Fourier series,  N = {N}")
    return line,


ani = animation.FuncAnimation(fig, update, frames=range(1, N_max+1),
                              interval=200, blit=False)

fig.subplots_adjust(top=0.9)
ani.save(f"{name}_N{N_max}.gif", writer="pillow", fps=5)   # 예: sawtooth_N50.gif
plt.show()
