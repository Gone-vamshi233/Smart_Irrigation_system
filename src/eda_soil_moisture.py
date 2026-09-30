import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SMART IRRIGATION - SOIL MOISTURE ANALYSIS")
print("=" * 60)

# Basic statistics
print("\nSoil Moisture Statistics:")
print(df["Soil_Moisture"].describe())


# Average soil moisture by irrigation need
print("\nAverage Soil Moisture by Irrigation Need:")
print(
    df.groupby("Irrigation_Need")["Soil_Moisture"]
      .mean()
      .sort_values()
)


# Boxplot by irrigation need
plt.figure(figsize=(8, 6))

df.boxplot(
    column="Soil_Moisture",
    by="Irrigation_Need"
)

plt.title("Soil Moisture vs Irrigation Need")
plt.suptitle("")
plt.xlabel("Irrigation Need")
plt.ylabel("Soil Moisture (%)")
plt.tight_layout()
plt.show()