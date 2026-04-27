import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje 
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# primijeni KMeans na RGB vrijednosti
km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)
km.fit(img_array)

# zamijeni svaki piksel s njemu najbližim centrom
labels = km.predict(img_array)
img_array_aprox = km.cluster_centers_[labels]

# pretvori nazad u oblik slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

# prikazi rezultantnu sliku
plt.figure()
plt.title("Kvantizirana slika (K=5)")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# lakat metoda
inertias = []
k_values = range(1, 11)
for k in k_values:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    km_temp.fit(img_array)
    inertias.append(km_temp.inertia_)

plt.figure()
plt.plot(k_values, inertias)
plt.xlabel('Broj grupa K')
plt.ylabel('J')
plt.title('Lakat metoda')
plt.show()