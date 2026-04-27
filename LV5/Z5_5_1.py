import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# generiranje podataka
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                          random_state=213, n_clusters_per_class=1, class_sep=1)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a) prikaz podataka
plt.figure()

#train podaci
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='train')

#test podaci
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='test')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Podaci")
plt.show()


# b) model logističke regresije
model = LogisticRegression()
model.fit(X_train, y_train)


# c) parametri i granica odluke
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

print("Parametri modela:")
print("theta0 =", theta0)
print("theta1 =", theta1)
print("theta2 =", theta2)

# granica odluke
x1_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_vals = -(theta0 + theta1 * x1_vals) / theta2

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.plot(x1_vals, x2_vals, color='red', label='granica odluke')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Granica odluke")
plt.show()


# d) klasifikacija
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("Matrica zabune:")
print(cm)
print("Točnost:", acc)
print("Preciznost:", prec)
print("Odziv:", rec)


# e) vizualizacija
correct = y_pred == y_test

plt.figure()

# točno
plt.scatter(X_test[correct][:, 0], X_test[correct][:, 1], color='green', label='točno')

# pogrešno
plt.scatter(X_test[~correct][:, 0], X_test[~correct][:, 1], color='black', label='pogrešno')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title("Rezultati na test skupu")
plt.show()