import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "raw" / "irrigation.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ml_ready_features.csv"

df = pd.read_csv(INPUT_PATH)


print("=" * 60)
print("SMART IRRIGATION - FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)


# --------------------------------------------------
# 2. Create engineered features
# --------------------------------------------------

# Soil moisture deficit
df["Moisture_Deficit"] = (
    100 - df["Soil_Moisture"]
).clip(lower=0)


# Previous water usage per hectare
df["Previous_Water_per_Hectare"] = (
    df["Previous_Irrigation_mm_24h"]
    / df["Field_Area_hectare"]
)


# Weather stress index
df["Weather_Stress_Index"] = (
    (df["Air_Temperature_C"] / 30)
    + ((100 - df["Relative_Humidity_pct"]) / 100)
    + (df["Wind_Speed_kmh"] / 20)
)


# Rainfall-adjusted moisture
df["Rainfall_Adjusted_Moisture"] = (
    df["Soil_Moisture"]
    + df["Rainfall_Last_24h_mm"]
)


# --------------------------------------------------
# 3. Remove tracking column from ML features
# --------------------------------------------------

if "Synthetic_Record" in df.columns:
    df = df.drop(columns=["Synthetic_Record"])


# --------------------------------------------------
# 4. Create processed directory
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 5. Save feature-engineered dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nFeature Engineering Completed.")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nEngineered Features:")
print(
    [
        "Moisture_Deficit",
        "Previous_Water_per_Hectare",
        "Weather_Stress_Index",
        "Rainfall_Adjusted_Moisture"
    ]
)

print("\nOutput File:")
print(OUTPUT_PATH)

print("\nFirst 5 Rows:")
print(df.head())

print("=" * 60)