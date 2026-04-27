import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)

# skaliraj
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

k_values = range(1, 51)
cv_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_n, y_train, cv=5)
    cv_scores.append(scores.mean())  

# ispisi optimalni K
best_k = k_values[np.argmax(cv_scores)]
print("Optimalni K: " + str(best_k))

# graf
plt.plot(k_values, cv_scores)
plt.xlabel('Broj susjeda K')
plt.ylabel('CV tocnost')
plt.show()