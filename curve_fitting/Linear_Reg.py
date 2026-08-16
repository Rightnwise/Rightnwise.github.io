import numpy as np
import matplotlib.pyplot as plt
n = 5
x_coords = []
y_coords = []


lim = 10
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks([t for t in ax.get_xticks() if t != 0])
ax.grid(True, alpha = 0.3)

mode = input("Choose mode: (1. Click points, 2. Enter points): ")
ax.set_title("Click points, then press Enter")
if mode == "1":
    pts = plt.ginput(n=-1, timeout = 0)
    x_coords = np.array([p[0] for p in pts])
    y_coords = np.array([p[1] for p in pts])
    n = len(x_coords)
else:
    n = int(input("Enter number of points: "))
    x_coords, y_coords = [], []
    for i in range(n):
        x_coords.append(float(input(f"x{i+1}:")))
        y_coords.append(float(input(f"y{i+1}:")))
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords) 

# y = ax + b
a = (n * np.sum(x_coords * y_coords)
        - np.sum(x_coords) * np.sum(y_coords)) / (n * np.sum(x_coords**2)
        - np.sum(x_coords)**2)
b = (-np.sum(x_coords) * np.sum(x_coords * y_coords)
        + np.sum(y_coords) * np.sum(x_coords**2)) / (n * np.sum(x_coords**2) - np.sum(x_coords)**2)

ax.scatter(x_coords, y_coords, color="blue")
xs = np.array([-lim, lim])
ax.plot(xs, a*xs + b, color = "red")
ax.set_title("Linear Regression")


plt.show()




