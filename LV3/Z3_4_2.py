import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

data["Fuel Type"] = data["Fuel Type"].astype("category")

#A)
data["CO2 Emissions (g/km)"].plot.hist(bins=30)
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frekvencija")
plt.title("Histogram CO2 emisija")
plt.show()

#B)
data.plot.scatter(
    x="Fuel Consumption City (L/100km)",
    y="CO2 Emissions (g/km)",
    c=data["Fuel Type"].cat.codes,
    cmap="jet"
)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("City vs CO2")
plt.show()

#C
data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Fuel Consumption Hwy (L/100km)")
plt.show()

#D)
cars_by_fuelType = data.groupby("Fuel Type").size()
cars_by_fuelType.plot(kind="bar")
plt.xlabel("Fuel Type")
plt.ylabel("Broj vozila")
plt.title("Broj vozila po tipu goriva")
plt.show()


#E)
cylinders_by_CO2 = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
cylinders_by_CO2.plot(kind="bar")
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("CO2 po broju cilindara")
plt.show()