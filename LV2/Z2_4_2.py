import numpy as np
import matplotlib.pyplot as plt

#a
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
print("Broj ljudi:", data.shape[0])

#b
plt.scatter(data[:,1], data[:,2])

plt.xlabel("Visina[cm]")
plt.ylabel("Masa[kg]")
plt.title("Zadatak 2.4.2 b)")
plt.show()

#c
plt.scatter(data[::50,1], data[::50,2])

plt.xlabel("Visina[cm]")
plt.ylabel("Masa[kg]")
plt.title("Zadatak 2.4.2 c)")
plt.show()

#d
print("\nMinimalna visina:", np.min(data[:,1]))
print("Maksimalna visina:", np.max(data[:,1]))
print("Srednja visina:", np.mean(data[:,1]))

#e
ind_m = (data[:,0] == 1)

print("\nMuškarci:")
print("Minimalna visina:", np.min(data[ind_m,1]))
print("Maksimalna visina:", np.max(data[ind_m,1]))
print("Srednja visina:", np.mean(data[ind_m,1]))


ind_z = (data[:,0] == 0)

print("\nŽene:")
print("Minimalna visina:", np.min(data[ind_z,1]))
print("Maksimalna visina:", np.max(data[ind_z,1]))
print("Srednja visina:", np.mean(data[ind_z,1]))