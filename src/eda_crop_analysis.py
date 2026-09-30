import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SMART IRRIGATION - CROP ANALYSIS")
print("=" * 60)


# Number of records for each crop
print("\nCrop Distribution:")
print(df["Crop_Type"].value_counts())


# Irrigation requirement distribution by crop
print("\nIrrigation Need by Crop:")

crop_irrigation = pd.crosstab(
    df["Crop_Type"],
    df["Irrigation_Need"]
)

print(crop_irrigation)


# Average soil moisture by crop
print("\nAverage Soil Moisture by Crop:")

crop_moisture = (
    df.groupby("Crop_Type")["Soil_Moisture"]
      .mean()
      .sort_values()
)

print(crop_moisture)


# Visualization
crop_irrigation.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Irrigation Need Distribution by Crop")
plt.xlabel("Crop Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.legend(title="Irrigation Need")
plt.tight_layout()
plt.show()