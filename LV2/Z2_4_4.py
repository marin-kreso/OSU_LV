import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")

#a
bright = np.clip(img * 1.5, 0, 1)

#b
quarter = img[:, img.shape[1]//4 : img.shape[1]//2]

#c
rotated = np.rot90(img, -1)

#d
mirror = img[:, ::-1]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0,0].imshow(bright)
axes[0,0].set_title("Posvijetljena")
axes[0,0].axis("off")

axes[0,1].imshow(quarter)
axes[0,1].set_title("Druga četvrtina")
axes[0,1].axis("off")

axes[1,0].imshow(rotated)
axes[1,0].set_title("Rotirana 90°")
axes[1,0].axis("off")

axes[1,1].imshow(mirror)
axes[1,1].set_title("Zrcaljena")
axes[1,1].axis("off")

plt.tight_layout()
plt.show()