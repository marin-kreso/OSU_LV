import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

data = pd.read_csv("data_C02_emission.csv")

#A)
X = data[[
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)"
]]

y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#B)
plt.scatter(X_train["Engine Size (L)"], y_train, color="blue", label="Train")
plt.scatter(X_test["Engine Size (L)"], y_test, color="red", label="Test")

plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Engine Size vs CO2 Emissions (g/km)")
plt.legend()
plt.show()

#C)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#prije i poslije skaliranja
plt.hist(X_train["Engine Size (L)"], alpha=0.5, label="Before scaling")
plt.hist(X_train_scaled[:, 0], alpha=0.5, label="After scaling")

plt.legend()
plt.show()

#D)
linearModel = LinearRegression()
linearModel.fit(X_train_scaled, y_train)

print("Koeficijenti:", linearModel.coef_)
print("Presjek:", linearModel.intercept_)

#E)
y_pred = linearModel.predict(X_test_scaled)

plt.scatter(y_test, y_pred)
plt.xlabel("Real CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Real vs Predicted Emissions")
plt.show()

#F)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

#G)
#Više ulaznih varijabli obično poboljšava model jer ima više informacija,
#ali previše varijabli može uzrokovati overfitting i lošije rezultate na testu.