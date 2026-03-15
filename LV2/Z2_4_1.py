import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 3, 2, 1, 3])
y = np.array([1, 2, 2, 1, 1])

plt.plot (x, y, "b", linewidth = 2, marker = "o", markersize = 7)
plt.axis([0, 4, 0, 4])
plt.xlabel("x-os")
plt.ylabel("y-os")
plt.title ("Zadatak 2.4.1")
plt.show()