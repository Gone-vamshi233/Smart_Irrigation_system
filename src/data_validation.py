import pandas as pd
from pathlib import Path


# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"

df = pd.read_csv(DATA_PATH)


# Define valid ranges for important sensor/weather features
range_checks = {
    "Soil_pH": (0, 14),
    "Soil_Moisture": (0, 100),
    "Relative_Humidity_pct": (0, 100),
    "Rainfall_Last_24h_mm": (0, None),
    "Sunlight_Hours": (0, 24),
    "Wind_Speed_kmh": (0, None),
    "Field_Area_hectare": (0, None),
    "Previous_Irrigation_mm_24h": (0, None),
    "Moisture_Deficit": (0, 100),
}


print("=" * 60)
print("SMART IRRIGATION - DATA VALIDATION")
print("=" * 60)

for column, (lower, upper) in range_checks.items():

    invalid = 0

    if lower is not None:
        invalid += (df[column] < lower).sum()

    if upper is not None:
        invalid += (df[column] > upper).sum()

    print(f"{column}: {invalid} invalid values")


print("=" * 60)
print("Data validation completed.")
print("=" * 60)