import pandas as pd

df = pd.read_csv("data_C02_emission.csv")

#A
print("Broj mjerenja:", len(df))
print("\nTipovi podataka:\n", df.dtypes)
print("\nIzostale vrijednosti:\n", df.isnull().sum())
print("\nBroj duplikata:", df.duplicated().sum())

df = df.dropna().drop_duplicates()

categorical_cols = ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]
df[categorical_cols] = df[categorical_cols].astype("category")

#B
sorted_df = df.sort_values(by="Fuel Consumption City (L/100km)")

print("\nNajmanja gradska potrošnja:")
print(sorted_df[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))

print("\nNajveća gradska potrošnja:")
print(sorted_df[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))

#C
subset = df[(df["Engine Size (L)"] >= 2.5) & (df["Engine Size (L)"] <= 3.5)]

print("\nBroj vozila(2.5–3.5L):", len(subset))
print("Prosječna CO2 emisija:", subset["CO2 Emissions (g/km)"].mean())

#D
audi = df[df["Make"] == "Audi"]
print("\nBroj Audi vozila:", len(audi))

audi_4 = audi[audi["Cylinders"] == 4]
print("Prosječna CO2 emisija (Audi, 4 cilindra):", audi_4["CO2 Emissions (g/km)"].mean())

#E
print("\nBroj vozila po cilindrima:")
print(df["Cylinders"].value_counts())

print("\nProsječna CO2 emisija po cilindrima:")
print(df.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())

#F
diesel = df[df["Fuel Type"] == "D"]
petrol = df[df["Fuel Type"] == "X"]

print("\nDizel prosjek:", diesel["Fuel Consumption City (L/100km)"].mean())
print("Dizel medijan:", diesel["Fuel Consumption City (L/100km)"].median())
print("\nBenzin prosjek:", petrol["Fuel Consumption City (L/100km)"].mean())
print("Benzin medijan:\n", petrol["Fuel Consumption City (L/100km)"].median())

#G
cars = df[(df["Fuel Type"] == "D") & (df["Cylinders"] == 4)]
wanted_car = cars.nlargest(1, "Fuel Consumption City (L/100km)")
print(wanted_car[["Make", "Model", "Fuel Consumption City (L/100km)"]])

#H
manual = df[df["Transmission"].str.contains("M")]
print("\nBroj vozila s ručnim mjenjačem:", len(manual))

#I
corr = df.corr(numeric_only=True)
print("\nKorelacijska matrica:\n", corr)