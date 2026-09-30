import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SMART IRRIGATION - CROP GROWTH STAGE ANALYSIS")
print("=" * 60)


# Check growth stages present in the dataset
print("\nCrop Growth Stages:")
print(df["Crop_Growth_Stage"].value_counts())


# Irrigation need distribution by growth stage
print("\nIrrigation Need by Growth Stage:")

growth_irrigation = pd.crosstab(
    df["Crop_Growth_Stage"],
    df["Irrigation_Need"]
)

print(growth_irrigation)


# Average soil moisture by growth stage
print("\nAverage Soil Moisture by Growth Stage:")

growth_moisture = (
    df.groupby("Crop_Growth_Stage")["Soil_Moisture"]
      .mean()
      .sort_values()
)

print(growth_moisture)


# Visualization
growth_irrigation.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Irrigation Need Distribution by Crop Growth Stage")
plt.xlabel("Crop Growth Stage")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.legend(title="Irrigation Need")
plt.tight_layout()
plt.show()