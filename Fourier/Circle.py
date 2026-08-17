# (2) 넓이는 r의 제곱에 비례함, r을 랜덤하게 뽑고 sqrt하면 균등해짐

import numpy as np
import matplotlib.pyplot as plt

n = 10000
theta = np.random.uniform(0, 1, n) * 2 * np.pi
r = (np.random.uniform(0, 1, n)**2)

x_coords = r * np.cos(theta)
y_coords = r * np.sin(theta)

plt.hist(r, bins = 15, color = 'skyblue', edgecolor = 'black')

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()
