import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SMART IRRIGATION - HISTORICAL IRRIGATION ANALYSIS")
print("=" * 60)


# Statistics of previous irrigation
print("\nPrevious Irrigation Statistics:")
print(
    df["Previous_Irrigation_mm_24h"].describe()
)


# Average previous irrigation by irrigation need
print("\nAverage Previous Irrigation by Irrigation Need:")

previous_irrigation = (
    df.groupby("Irrigation_Need")["Previous_Irrigation_mm_24h"]
      .mean()
      .sort_values()
)

print(previous_irrigation)


# Average water usage per hectare by irrigation need
print("\nAverage Previous Water per Hectare by Irrigation Need:")

water_per_hectare = (
    df.groupby("Irrigation_Need")["Previous_Water_per_Hectare"]
      .mean()
      .sort_values()
)

print(water_per_hectare)


# Boxplot: Previous irrigation vs irrigation need
plt.figure(figsize=(8, 6))

df.boxplot(
    column="Previous_Irrigation_mm_24h",
    by="Irrigation_Need"
)

plt.title("Previous Irrigation vs Current Irrigation Need")
plt.suptitle("")
plt.xlabel("Current Irrigation Need")
plt.ylabel("Previous Irrigation (mm)")
plt.tight_layout()
plt.show()