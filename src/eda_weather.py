import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SMART IRRIGATION - WEATHER & RAINFALL ANALYSIS")
print("=" * 60)


# Weather statistics
weather_columns = [
    "Air_Temperature_C",
    "Relative_Humidity_pct",
    "Rainfall_Last_24h_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh"
]

print("\nWeather Statistics:")
print(df[weather_columns].describe())


# Average rainfall by irrigation need
print("\nAverage Rainfall by Irrigation Need:")
print(
    df.groupby("Irrigation_Need")["Rainfall_Last_24h_mm"]
      .mean()
      .sort_values()
)


# Average temperature by irrigation need
print("\nAverage Temperature by Irrigation Need:")
print(
    df.groupby("Irrigation_Need")["Air_Temperature_C"]
      .mean()
      .sort_values()
)


# Rainfall vs irrigation need
plt.figure(figsize=(8, 6))

df.boxplot(
    column="Rainfall_Last_24h_mm",
    by="Irrigation_Need"
)

plt.title("Rainfall vs Irrigation Need")
plt.suptitle("")
plt.xlabel("Irrigation Need")
plt.ylabel("Rainfall in Last 24 Hours (mm)")
plt.tight_layout()
plt.show()