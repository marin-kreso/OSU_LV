import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error

data = pd.read_csv("data_C02_emission.csv")

X = data[[
        "Engine Size (L)",
        "Cylinders",
        "Fuel Consumption City (L/100km)",
        "Fuel Consumption Hwy (L/100km)",
        "Fuel Consumption Comb (L/100km)",
        "Fuel Type"
    ]
]

y = data["CO2 Emissions (g/km)"]

X = pd.get_dummies(X, columns=["Fuel Type"], drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

max_err = max_error(y_test, y_pred)
print("Maksimalna pogreška (g/km):", max_err)

errors = abs(y_test - y_pred)
worst_index = errors.idxmax()

print("\nModel vozila s najvećom pogreškom:")
print(data.loc[worst_index, ["Make", "Model", "Vehicle Class", "Fuel Type"]])
print("Stvarna CO2:", y_test.loc[worst_index])
print("Predviđena CO2:", y_pred[list(y_test.index).index(worst_index)])